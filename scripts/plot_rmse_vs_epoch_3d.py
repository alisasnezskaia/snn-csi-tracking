"""3D twin of plot_rmse_vs_epoch.py -- same leave-activity-out CV protocol,
same three models (SNN best cross_coherence config, ANN twin, LSTM), same
cross_coherence feature set, but out_dim=3 (X, Y, Z) instead of 2. (X, Y, Z)
is real camera-relative position in meters, from deprojecting the MediaPipe
head pixel through Depth-Anything-V2's metric-indoor depth at that point --
see camera_geometry.deproject_pixel_to_camera_frame /
presence_position_dataset.resample_depth. RMSE here is in meters (camera
frame, not room-referenced -- see that function's docstring for why that's
a legitimate simplification for this fixed-camera dataset).

Outputs are named distinctly from the 2D run (rmse_vs_epoch_3d.* /
rmse_vs_epoch_history_3d.npz) so the two can be compared side by side
without overwriting each other.

Run:
    .venv/bin/python scripts/plot_rmse_vs_epoch_3d.py
Output:
    results/figures/rmse_vs_epoch_3d.png (and .pdf)
    results/rmse_vs_epoch_history_3d.npz
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from snn_csi_tracking.data.dataset import CSIDataset
from snn_csi_tracking.data.presence_position_dataset import load_or_build_perframe_dataset
from snn_csi_tracking.models.presence_position_ann import ANNPresencePositionConvPerFrame
from snn_csi_tracking.models.presence_position_lstm import LSTMPresencePositionConvPerFrame
from snn_csi_tracking.models.presence_position_snn import SNNPresencePositionConvPerFrame
from snn_csi_tracking.training.train_presence_position_conv import (
    AMPLITUDE_NORM, BATCH_SIZE, CACHE_DIR, CONV_CHANNELS, DELTA_THRESHOLD, DEPTH_CACHE_DIR, HIDDEN_1,
    HIDDEN_2, KERNEL_SIZE, LEARNING_RATE, NUM_EPOCHS, NUM_WORKERS, RATE_MS, RAW_ROOT, SMOOTHNESS_WEIGHT,
    STRIDE, T_WIN, TRAJECTORY_CACHE_DIR, USE_EMPTY_BASELINE, USE_MOTION_MAGNITUDE, USE_PHASE,
    collect_predictions, position_norm_stats, session_key, train_one_epoch,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONDITIONS = ("NLoS", "PLoS")
ROTATION = ["EN-S", "EN-W", "L", "S"]  # E excluded -- always kept in train
OUT_DIM = 3

MODELS = ["snn", "ann", "lstm"]
LABELS = {"snn": "SNN (best, cross\\_coherence)", "ann": "ANN", "lstm": "LSTM"}
COLORS = {"snn": "#2563eb", "ann": "#dc2626", "lstm": "#eab308"}  # same convention as the 2D plot


def make_fold_loaders(X, pos, present, groups, test_activity: str, val_activity: str):
    sessions = np.array([session_key(g) for g in groups])
    test_sessions = {f"{c}_{test_activity}" for c in CONDITIONS}
    val_sessions = {f"{c}_{val_activity}" for c in CONDITIONS}
    bucket = np.array([
        "test" if s in test_sessions else "val" if s in val_sessions else "train" for s in sessions
    ])
    train_idx, val_idx, test_idx = np.where(bucket == "train")[0], np.where(bucket == "val")[0], np.where(bucket == "test")[0]
    train_ds = CSIDataset(X, pos, train_idx, label_dtype=torch.float32, extra=present)
    val_ds = CSIDataset(X, pos, val_idx, label_dtype=torch.float32, extra=present)
    test_ds = CSIDataset(X, pos, test_idx, label_dtype=torch.float32, extra=present)
    loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), persistent_workers=NUM_WORKERS > 0)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs),
        train_ds,
    )


def build_model(model_type: str, num_channels: int, num_subcarriers: int) -> nn.Module:
    common = dict(num_channels=num_channels, num_subcarriers=num_subcarriers, conv_channels=CONV_CHANNELS,
                  h1=HIDDEN_1, h2=HIDDEN_2, out_dim=OUT_DIM, kernel_size=KERNEL_SIZE)
    if model_type == "snn":
        return SNNPresencePositionConvPerFrame(**common, delta_threshold=DELTA_THRESHOLD, encoder_type="perframe").to(DEVICE)
    if model_type == "ann":
        return ANNPresencePositionConvPerFrame(**common).to(DEVICE)
    return LSTMPresencePositionConvPerFrame(**common).to(DEVICE)


def evaluate_full_and_xy(model, val_loader, mu, sigma) -> tuple[float, float]:
    """One forward pass over val_loader, two RMSEs from it: the full xyz
    (sqrt(sum of squared error over all 3 dims), same as evaluate()'s own
    rmse_and_presence_acc) and an xy-only version that drops the z (depth)
    term entirely -- lets the 3D-trained model's x,y accuracy be compared
    directly against the 2D-only run, isolating whether adding z actually
    hurt x,y or whether the combined-metric gap is mostly z being a
    noisier target. pos_pred comes back from the model in normalized
    (z-scored) space (see position_norm_stats) -- un-normalized here before
    computing either RMSE, so both stay in real meters."""
    _pres_pred, pos_pred, pres, pos = collect_predictions(model, val_loader)
    pos_pred = pos_pred * sigma + mu
    sq_err_full = ((pos_pred - pos) ** 2).sum(dim=-1)
    rmse_full = torch.sqrt((sq_err_full * pres).sum() / (pres.sum() + 1e-8)).item()
    sq_err_xy = ((pos_pred[..., :2] - pos[..., :2]) ** 2).sum(dim=-1)
    rmse_xy = torch.sqrt((sq_err_xy * pres).sum() / (pres.sum() + 1e-8)).item()
    return rmse_full, rmse_xy


def run_fold(model_type: str, X, pos, present, groups, test_activity: str, val_activity: str) -> tuple[np.ndarray, np.ndarray]:
    """Trains one fold for NUM_EPOCHS, returns (per-epoch full-xyz val_rmse, per-epoch xy-only val_rmse)."""
    train_loader, val_loader, _test_loader, train_ds = make_fold_loaders(X, pos, present, groups, test_activity, val_activity)
    train_presence_rate = train_ds.extra.float().mean().item()
    pos_weight = torch.tensor([(1 - train_presence_rate) / train_presence_rate]).to(DEVICE)
    mu, sigma = position_norm_stats(train_ds)

    model = build_model(model_type, X.shape[1], X.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    rmse_history = np.zeros(NUM_EPOCHS)
    rmse_xy_history = np.zeros(NUM_EPOCHS)
    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model, train_loader, optimizer, bce_loss, SMOOTHNESS_WEIGHT, mu, sigma)
        rmse_history[epoch], rmse_xy_history[epoch] = evaluate_full_and_xy(model, val_loader, mu, sigma)
    return rmse_history, rmse_xy_history


def main():
    print(f"Loading 3D dataset (amplitude_norm={AMPLITUDE_NORM}, feature=cross_coherence, out_dim={OUT_DIM}, "
          f"z=metric depth, camera-frame meters)...")
    X, pos, present, groups, _activity_codes = load_or_build_perframe_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, DEPTH_CACHE_DIR, CACHE_DIR,
        rate_ms=RATE_MS, t_win=T_WIN, stride=STRIDE, use_3d=True, use_phase=USE_PHASE,
        use_empty_baseline=USE_EMPTY_BASELINE, denoise=None, use_motion_magnitude=USE_MOTION_MAGNITUDE,
        amplitude_norm=AMPLITUDE_NORM, use_relative_motion=False, use_cross_coherence=True,
    )
    X = np.asarray(X)
    print(f"dataset shape: X={X.shape}  pos={pos.shape}")

    history = {m: np.zeros((len(ROTATION), NUM_EPOCHS)) for m in MODELS}
    history_xy = {m: np.zeros((len(ROTATION), NUM_EPOCHS)) for m in MODELS}
    t_start = time.time()
    for model_type in MODELS:
        for i, test_activity in enumerate(ROTATION):
            val_activity = ROTATION[(i + 1) % len(ROTATION)]
            t0 = time.time()
            history[model_type][i], history_xy[model_type][i] = run_fold(
                model_type, X, pos, present, groups, test_activity, val_activity
            )
            print(f"{model_type:4s} fold {i+1}/{len(ROTATION)} (test={test_activity}, val={val_activity}): "
                  f"final val_rmse_xyz={history[model_type][i, -1]:.4f}  "
                  f"final val_rmse_xy={history_xy[model_type][i, -1]:.4f}  ({time.time()-t0:.1f}s)")
    print(f"total wall time: {time.time()-t_start:.1f}s")

    out_dir = REPO_ROOT / "results"
    np.savez(out_dir / "rmse_vs_epoch_history_3d.npz", epochs=np.arange(1, NUM_EPOCHS + 1),
             **{f"{m}_rmse": history[m] for m in MODELS},
             **{f"{m}_rmse_xy": history_xy[m] for m in MODELS})
    print(f"saved raw per-fold history (xyz + xy-only) to {out_dir / 'rmse_vs_epoch_history_3d.npz'}")

    print("\n=== final-epoch full xyz RMSE (meters, camera frame) ===")
    for m in MODELS:
        xyz_final = history[m][:, -1]
        print(f"{m:4s}  {xyz_final.mean():.4f} +/- {xyz_final.std():.4f} m")
    # No xy-only-vs-2D-only comparison here anymore: the 3D model's (x,y) is now
    # real camera-frame meters (deproject_pixel_to_camera_frame), while the 2D-only
    # run's (x,y) is still normalized [0,1] image-plane -- the two are no longer on
    # a shared scale, so that comparison would silently mix units.

    plot(history, out_dir / "figures")


def plot(history: dict[str, np.ndarray], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 9,
    })

    epochs = np.arange(1, NUM_EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))

    for model_type in MODELS:
        rmse = history[model_type]  # (n_folds, n_epochs)
        mean, std = rmse.mean(axis=0), rmse.std(axis=0)
        color = COLORS[model_type]
        ax.plot(epochs, mean, label=LABELS[model_type], color=color, linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE (m, camera frame)")
    ax.set_xlim(1, NUM_EPOCHS)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, which="major")
    ax.legend(frameon=False)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "rmse_vs_epoch_3d.png", dpi=200)
    plt.savefig(out_dir / "rmse_vs_epoch_3d.pdf")
    print(f"saved to {out_dir / 'rmse_vs_epoch_3d.png'} and .pdf")


if __name__ == "__main__":
    main()
