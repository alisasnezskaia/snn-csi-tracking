"""RMSE-vs-epoch training-curve comparison for our best SNN config (feature=
cross_coherence, see train_presence_position_cv.py) against its two
conventional-DL twins -- the ANN (models.presence_position_ann, same
leaky-accumulation structure minus spiking) and the LSTM
(models.presence_position_lstm, real 2-layer nn.LSTM recurrence, see that
module's docstring for why it's the fairer conventional benchmark).

Same leave-activity-out CV protocol as train_presence_position_cv.py /
train_presence_position_cv_lstm.py (4 folds, one per non-empty-room
activity). None of those scripts persist the per-epoch val_rmse they
already compute each epoch (train_one_epoch/evaluate) -- they print it and
move on. This script is the same training loop, just keeping that number
instead of discarding it, averaged across folds per epoch so a single
fold's noise doesn't dominate the curve.

Run:
    .venv/bin/python scripts/plot_rmse_vs_epoch.py
Output:
    results/figures/rmse_vs_epoch.png (and .pdf)
    results/rmse_vs_epoch_history.npz  (raw per-fold, per-epoch RMSE, so the
    plot can be regenerated without re-training)
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
    collect_predictions, identity_norm_stats, presence_balanced_acc, rmse_and_presence_acc, session_key,
    train_one_epoch,
)
from train_presence_position_cv import presence_auroc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONDITIONS = ("NLoS", "PLoS")
ROTATION = ["EN-S", "EN-W", "L", "S"]  # E excluded -- always kept in train

MODELS = ["snn", "ann", "lstm"]
LABELS = {"snn": "SNN (best, cross\\_coherence)", "ann": "ANN", "lstm": "LSTM"}
COLORS = {"snn": "#2563eb", "ann": "#dc2626", "lstm": "#eab308"}  # blue/red/yellow -- blue/purple were
# too easy to confuse (esp. protanomaly/deuteranomaly), checked with colorspacious CVD simulation


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
                  h1=HIDDEN_1, h2=HIDDEN_2, out_dim=2, kernel_size=KERNEL_SIZE)
    if model_type == "snn":
        return SNNPresencePositionConvPerFrame(**common, delta_threshold=DELTA_THRESHOLD, encoder_type="perframe").to(DEVICE)
    if model_type == "ann":
        return ANNPresencePositionConvPerFrame(**common).to(DEVICE)
    return LSTMPresencePositionConvPerFrame(**common).to(DEVICE)


def run_fold(model_type: str, X, pos, present, groups, test_activity: str, val_activity: str
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trains one fold for NUM_EPOCHS, returns per-epoch (val_rmse, val_bacc, val_auroc), each (NUM_EPOCHS,).
    bacc (balanced accuracy, fixed threshold=0.5) and auroc (threshold-independent) are the SAME presence
    metrics train_presence_position_cv.py uses for checkpoint selection / reporting -- tracking them per
    epoch here (not just RMSE) is what lets a presence-learning collapse (stuck near chance) actually show
    up in this curve, instead of silently going unmeasured."""
    train_loader, val_loader, _test_loader, train_ds = make_fold_loaders(X, pos, present, groups, test_activity, val_activity)
    train_presence_rate = train_ds.extra.float().mean().item()
    pos_weight = torch.tensor([(1 - train_presence_rate) / train_presence_rate]).to(DEVICE)
    mu, sigma = identity_norm_stats(2)  # 2D (x,y) is already [0,1] by construction -- see identity_norm_stats

    model = build_model(model_type, X.shape[1], X.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    rmse_history = np.zeros(NUM_EPOCHS)
    bacc_history = np.zeros(NUM_EPOCHS)
    auroc_history = np.zeros(NUM_EPOCHS)
    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model, train_loader, optimizer, bce_loss, SMOOTHNESS_WEIGHT, mu, sigma)
        pres_pred, pos_pred, pres_gt, pos_gt = collect_predictions(model, val_loader)
        rmse_history[epoch], _presence_acc = rmse_and_presence_acc(pres_pred, pos_pred, pres_gt, pos_gt, mu=mu, sigma=sigma)
        bacc_history[epoch] = presence_balanced_acc(pres_pred, pres_gt, threshold=0.5)
        auroc_history[epoch] = presence_auroc(pres_pred, pres_gt)
    return rmse_history, bacc_history, auroc_history


def main():
    print(f"Loading dataset (amplitude_norm={AMPLITUDE_NORM}, feature=cross_coherence, "
          f"matching our best SNN config)...")
    X, pos, present, groups, _activity_codes = load_or_build_perframe_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, DEPTH_CACHE_DIR, CACHE_DIR,
        rate_ms=RATE_MS, t_win=T_WIN, stride=STRIDE, use_3d=False, use_phase=USE_PHASE,
        use_empty_baseline=USE_EMPTY_BASELINE, denoise=None, use_motion_magnitude=USE_MOTION_MAGNITUDE,
        amplitude_norm=AMPLITUDE_NORM, use_relative_motion=False, use_cross_coherence=True,
    )
    X = np.asarray(X)

    history = {m: np.zeros((len(ROTATION), NUM_EPOCHS)) for m in MODELS}
    history_bacc = {m: np.zeros((len(ROTATION), NUM_EPOCHS)) for m in MODELS}
    history_auroc = {m: np.zeros((len(ROTATION), NUM_EPOCHS)) for m in MODELS}
    t_start = time.time()
    for model_type in MODELS:
        for i, test_activity in enumerate(ROTATION):
            val_activity = ROTATION[(i + 1) % len(ROTATION)]
            t0 = time.time()
            history[model_type][i], history_bacc[model_type][i], history_auroc[model_type][i] = run_fold(
                model_type, X, pos, present, groups, test_activity, val_activity
            )
            print(f"{model_type:4s} fold {i+1}/{len(ROTATION)} (test={test_activity}, val={val_activity}): "
                  f"final val_rmse={history[model_type][i, -1]:.4f}  final val_bacc={history_bacc[model_type][i, -1]:.3f}  "
                  f"final val_auroc={history_auroc[model_type][i, -1]:.3f}  ({time.time()-t0:.1f}s)")
    print(f"total wall time: {time.time()-t_start:.1f}s")

    out_dir = REPO_ROOT / "results"
    np.savez(out_dir / "rmse_vs_epoch_history.npz", epochs=np.arange(1, NUM_EPOCHS + 1),
             **{f"{m}_rmse": history[m] for m in MODELS},
             **{f"{m}_bacc": history_bacc[m] for m in MODELS},
             **{f"{m}_auroc": history_auroc[m] for m in MODELS})
    print(f"saved raw per-fold history (rmse + bacc + auroc) to {out_dir / 'rmse_vs_epoch_history.npz'}")

    plot(history, history_bacc, out_dir / "figures")


def plot(history: dict[str, np.ndarray], history_bacc: dict[str, np.ndarray], out_dir: Path):
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
    fig, (ax_rmse, ax_bacc) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    for model_type in MODELS:
        rmse = history[model_type]  # (n_folds, n_epochs)
        mean, std = rmse.mean(axis=0), rmse.std(axis=0)
        color = COLORS[model_type]
        ax_rmse.plot(epochs, mean, label=LABELS[model_type], color=color, linewidth=2)
        ax_rmse.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    ax_rmse.set_xlabel("Epoch")
    ax_rmse.set_ylabel("Validation RMSE (normalized position)")
    ax_rmse.set_xlim(1, NUM_EPOCHS)
    ax_rmse.set_ylim(bottom=0)
    ax_rmse.spines["top"].set_visible(False)
    ax_rmse.spines["right"].set_visible(False)
    ax_rmse.grid(axis="y", alpha=0.25, which="major")
    ax_rmse.legend(frameon=False)

    for model_type in MODELS:
        bacc = history_bacc[model_type]  # (n_folds, n_epochs)
        mean, std = bacc.mean(axis=0), bacc.std(axis=0)
        color = COLORS[model_type]
        ax_bacc.plot(epochs, mean, label=LABELS[model_type], color=color, linewidth=2)
        ax_bacc.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    ax_bacc.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)  # chance level

    ax_bacc.set_xlabel("Epoch")
    ax_bacc.set_ylabel("Validation presence balanced accuracy")
    ax_bacc.set_xlim(1, NUM_EPOCHS)
    ax_bacc.set_ylim(0, 1)
    ax_bacc.spines["top"].set_visible(False)
    ax_bacc.spines["right"].set_visible(False)
    ax_bacc.grid(axis="y", alpha=0.25, which="major")
    ax_bacc.legend(frameon=False)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "rmse_vs_epoch.png", dpi=200)
    plt.savefig(out_dir / "rmse_vs_epoch.pdf")
    print(f"saved to {out_dir / 'rmse_vs_epoch.png'} and .pdf")


if __name__ == "__main__":
    main()
