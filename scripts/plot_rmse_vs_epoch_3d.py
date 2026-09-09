"""3D twin of plot_rmse_vs_epoch.py -- same leave-activity-out CV protocol,
same three models (SNN best cross_coherence config, ANN twin, LSTM), same
cross_coherence feature set, but out_dim=3 (X, Y, Z) instead of 2.

Two --pinhole modes (see presence_position_dataset.build_perframe_dataset):
  --pinhole (default): (X, Y, Z) is real camera-relative position in
    meters, from deprojecting the MediaPipe head pixel through
    Depth-Anything-V2's metric-indoor depth at that point -- see
    camera_geometry.deproject_pixel_to_camera_frame. RMSE here is in
    meters (camera frame, not room-referenced -- see that function's
    docstring for why that's a legitimate simplification for this
    fixed-camera dataset) and is NOT on the same scale as the 2D run.
  --no-pinhole: (x, y) stay the same normalized [0,1] image-plane values
    as the 2D run, with z = metric depth min-max normalized to [0,1] over
    the whole dataset. RMSE here stays on the same [0,1] scale as the 2D
    run, so it's the one actually comparable to it.

Outputs are named distinctly per mode (rmse_vs_epoch_3d.* /
rmse_vs_epoch_history_3d.npz for --pinhole, rmse_vs_epoch_3dz.* /
rmse_vs_epoch_history_3dz.npz for --no-pinhole) so none of the three runs
(2D, 3D pinhole, 3D no-pinhole) overwrite each other.

Run:
    .venv/bin/python scripts/plot_rmse_vs_epoch_3d.py
    .venv/bin/python scripts/plot_rmse_vs_epoch_3d.py --no-pinhole
Output:
    results/figures/rmse_vs_epoch_3d{,z}.png (and .pdf)
    results/rmse_vs_epoch_history_3d{,z}.npz
"""

from __future__ import annotations

import argparse
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
    collect_predictions, identity_norm_stats, position_norm_stats, presence_balanced_acc, session_key, train_one_epoch,
)
from train_presence_position_cv import presence_auroc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONDITIONS = ("NLoS", "PLoS")
ROTATION = ["EN-S", "EN-W", "L", "S"]  # E excluded -- always kept in train
OUT_DIM = 3

MODELS = ["snn", "ann", "lstm"]
LABELS = {"snn": "SNN (best, cross\\_coherence)", "ann": "ANN", "lstm": "LSTM"}
COLORS = {"snn": "#2563eb", "ann": "#dc2626", "lstm": "#eab308"}  # same convention as the 2D plot

# --feature full adds raw motion magnitude, relative (normalized) motion, spectral
# ratio, AND cross-coherence together -- the actual Eq. (9) F[t] feature stack from
# the paper, vs. the historical --feature cross_coherence default (AP+CC only, the
# config every existing results/rmse_vs_epoch_history_3d.npz and
# results/figures/rmse_vs_epoch_3d.{pdf,png} were produced with). Writes to
# differently-named outputs (see out_tag below) so a --feature full run never
# overwrites the existing AP+CC results -- both can be compared side by side.
FEATURE_KWARGS = {
    "cross_coherence": dict(use_motion_magnitude=USE_MOTION_MAGNITUDE, use_relative_motion=False,
                             use_spectral_ratio=False, use_cross_coherence=True),
    "full": dict(use_motion_magnitude=True, use_relative_motion=True,
                 use_spectral_ratio=True, use_cross_coherence=True),
}


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


def evaluate_epoch(model, val_loader, mu, sigma) -> tuple[float, float, float, float]:
    """One forward pass over val_loader, four per-epoch metrics from it:
    the full xyz RMSE (sqrt(sum of squared error over all 3 dims), same as
    evaluate()'s own rmse_and_presence_acc), an xy-only RMSE that drops the
    z (depth) term entirely -- lets the 3D-trained model's x,y accuracy be
    compared directly against the 2D-only run, isolating whether adding z
    actually hurt x,y or whether the combined-metric gap is mostly z being
    a noisier target (in --pinhole mode this mixes units against the 2D
    run's [0,1] scale; in --no-pinhole mode both are the same [0,1]
    image-plane scale, so the comparison is literal) -- and presence
    balanced accuracy (fixed threshold=0.5) + AUROC, the SAME presence
    metrics train_presence_position_cv.py tracks, computed from the same
    collect_predictions call so tracking presence per epoch doesn't cost a
    second val forward pass. pos_pred comes back from the model in
    normalized space (min-max via position_norm_stats for --pinhole; a
    no-op via identity_norm_stats for --no-pinhole, see run_fold) --
    un-normalized here before computing either RMSE, so both stay in the
    same real units as the loaded dataset."""
    pres_pred, pos_pred, pres, pos = collect_predictions(model, val_loader)
    pos_pred = pos_pred * sigma + mu
    sq_err_full = ((pos_pred - pos) ** 2).sum(dim=-1)
    rmse_full = torch.sqrt((sq_err_full * pres).sum() / (pres.sum() + 1e-8)).item()
    sq_err_xy = ((pos_pred[..., :2] - pos[..., :2]) ** 2).sum(dim=-1)
    rmse_xy = torch.sqrt((sq_err_xy * pres).sum() / (pres.sum() + 1e-8)).item()
    bacc = presence_balanced_acc(pres_pred, pres, threshold=0.5)
    auroc = presence_auroc(pres_pred, pres)
    return rmse_full, rmse_xy, bacc, auroc


def run_fold(model_type: str, X, pos, present, groups, test_activity: str, val_activity: str, pinhole: bool = True
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trains one fold for NUM_EPOCHS, returns per-epoch (full-xyz val_rmse, xy-only val_rmse,
    val_bacc, val_auroc), each (NUM_EPOCHS,)."""
    train_loader, val_loader, _test_loader, train_ds = make_fold_loaders(X, pos, present, groups, test_activity, val_activity)
    train_presence_rate = train_ds.extra.float().mean().item()
    pos_weight = torch.tensor([(1 - train_presence_rate) / train_presence_rate]).to(DEVICE)
    # position_norm_stats only earns its keep for real-meters pinhole positions; --no-pinhole
    # is already [0,1] by construction (see identity_norm_stats).
    mu, sigma = position_norm_stats(train_ds) if pinhole else identity_norm_stats(OUT_DIM)

    model = build_model(model_type, X.shape[1], X.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    rmse_history = np.zeros(NUM_EPOCHS)
    rmse_xy_history = np.zeros(NUM_EPOCHS)
    bacc_history = np.zeros(NUM_EPOCHS)
    auroc_history = np.zeros(NUM_EPOCHS)
    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model, train_loader, optimizer, bce_loss, SMOOTHNESS_WEIGHT, mu, sigma)
        rmse_history[epoch], rmse_xy_history[epoch], bacc_history[epoch], auroc_history[epoch] = evaluate_epoch(
            model, val_loader, mu, sigma
        )
    return rmse_history, rmse_xy_history, bacc_history, auroc_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pinhole", dest="pinhole", action="store_false", default=True,
                         help="skip the pinhole deprojection and keep (x, y) as normalized [0,1] image-plane "
                              "values with z = metric depth min-max normalized to [0,1] over the whole dataset "
                              "-- RMSE then stays on the same [0,1] scale as the 2D-only run")
    parser.add_argument("--feature", choices=["cross_coherence", "full"], default="cross_coherence",
                         help="cross_coherence (default): AP+CC only, matches every existing results/ output. "
                              "full: the complete Eq. (9) feature stack (raw motion magnitude + relative motion "
                              "+ spectral ratio + cross-coherence together) -- writes to separately-named "
                              "outputs, never overwrites the cross_coherence results.")
    args = parser.parse_args()
    unit_desc = "camera-frame meters" if args.pinhole else "[0,1] image-plane x,y + [0,1]-normalized depth z"
    out_tag = ("3d" if args.pinhole else "3dz") + ("" if args.feature == "cross_coherence" else "_full")
    if args.feature == "full":
        LABELS["snn"] = "SNN (full feature vector)"

    print(f"Loading 3D dataset (amplitude_norm={AMPLITUDE_NORM}, feature={args.feature}, out_dim={OUT_DIM}, "
          f"pinhole={args.pinhole}, z={unit_desc})...")
    X, pos, present, groups, _activity_codes = load_or_build_perframe_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, DEPTH_CACHE_DIR, CACHE_DIR,
        rate_ms=RATE_MS, t_win=T_WIN, stride=STRIDE, use_3d=True, pinhole=args.pinhole, use_phase=USE_PHASE,
        use_empty_baseline=USE_EMPTY_BASELINE, denoise=None, amplitude_norm=AMPLITUDE_NORM,
        **FEATURE_KWARGS[args.feature],
    )
    X = np.asarray(X)
    print(f"dataset shape: X={X.shape}  pos={pos.shape}")

    history = {m: np.zeros((len(ROTATION), NUM_EPOCHS)) for m in MODELS}
    history_xy = {m: np.zeros((len(ROTATION), NUM_EPOCHS)) for m in MODELS}
    history_bacc = {m: np.zeros((len(ROTATION), NUM_EPOCHS)) for m in MODELS}
    history_auroc = {m: np.zeros((len(ROTATION), NUM_EPOCHS)) for m in MODELS}
    t_start = time.time()
    for model_type in MODELS:
        for i, test_activity in enumerate(ROTATION):
            val_activity = ROTATION[(i + 1) % len(ROTATION)]
            t0 = time.time()
            history[model_type][i], history_xy[model_type][i], history_bacc[model_type][i], history_auroc[model_type][i] = run_fold(
                model_type, X, pos, present, groups, test_activity, val_activity, pinhole=args.pinhole
            )
            print(f"{model_type:4s} fold {i+1}/{len(ROTATION)} (test={test_activity}, val={val_activity}): "
                  f"final val_rmse_xyz={history[model_type][i, -1]:.4f}  "
                  f"final val_rmse_xy={history_xy[model_type][i, -1]:.4f}  "
                  f"final val_bacc={history_bacc[model_type][i, -1]:.3f}  "
                  f"final val_auroc={history_auroc[model_type][i, -1]:.3f}  ({time.time()-t0:.1f}s)")
    print(f"total wall time: {time.time()-t_start:.1f}s")

    out_dir = REPO_ROOT / "results"
    history_path = out_dir / f"rmse_vs_epoch_history_{out_tag}.npz"
    np.savez(history_path, epochs=np.arange(1, NUM_EPOCHS + 1),
             **{f"{m}_rmse": history[m] for m in MODELS},
             **{f"{m}_rmse_xy": history_xy[m] for m in MODELS},
             **{f"{m}_bacc": history_bacc[m] for m in MODELS},
             **{f"{m}_auroc": history_auroc[m] for m in MODELS})
    print(f"saved raw per-fold history (xyz + xy-only + bacc + auroc) to {history_path}")

    print(f"\n=== final-epoch full xyz RMSE ({unit_desc}) ===")
    for m in MODELS:
        xyz_final = history[m][:, -1]
        unit_suffix = " m" if args.pinhole else ""
        print(f"{m:4s}  {xyz_final.mean():.4f} +/- {xyz_final.std():.4f}{unit_suffix}")
    if args.pinhole:
        # No xy-only-vs-2D-only comparison here: the 3D model's (x,y) is real
        # camera-frame meters (deproject_pixel_to_camera_frame), while the 2D-only
        # run's (x,y) is still normalized [0,1] image-plane -- the two are not on
        # a shared scale, so that comparison would silently mix units.
        pass
    else:
        print("(--no-pinhole: rmse_xy above is directly comparable to the 2D-only run's RMSE -- same [0,1] scale)")

    print("\n=== final-epoch presence balanced accuracy / AUROC ===")
    print("(nan-aware: a fold whose val session is single-class the whole way through -- e.g. test=L,val=S --")
    print(" has bacc/auroc undefined for EVERY epoch, see presence_balanced_acc/presence_auroc; excluded from")
    print(" the mean/std below via nanmean/nanstd rather than poisoning it to an all-nan result)")
    for m in MODELS:
        bacc_final, auroc_final = history_bacc[m][:, -1], history_auroc[m][:, -1]
        n_valid = (~np.isnan(bacc_final)).sum()
        print(f"{m:4s}  bacc={np.nanmean(bacc_final):.3f} +/- {np.nanstd(bacc_final):.3f}   "
              f"auroc={np.nanmean(auroc_final):.3f} +/- {np.nanstd(auroc_final):.3f}   "
              f"({n_valid}/{len(bacc_final)} folds valid)")

    plot(history, history_bacc, out_dir / "figures", pinhole=args.pinhole, out_tag=out_tag)


def plot(history: dict[str, np.ndarray], history_bacc: dict[str, np.ndarray], out_dir: Path, pinhole: bool, out_tag: str):
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
    ylabel = "Validation RMSE (m, camera frame)" if pinhole else "Validation RMSE ([0,1], image-plane + normalized depth)"
    ax_rmse.set_ylabel(ylabel)
    ax_rmse.set_xlim(1, NUM_EPOCHS)
    ax_rmse.set_ylim(bottom=0)
    ax_rmse.spines["top"].set_visible(False)
    ax_rmse.spines["right"].set_visible(False)
    ax_rmse.grid(axis="y", alpha=0.25, which="major")
    ax_rmse.legend(frameon=False)

    for model_type in MODELS:
        bacc = history_bacc[model_type]  # (n_folds, n_epochs)
        # nanmean/nanstd: a fold whose val session is single-class the whole way through
        # (e.g. test=L,val=S) is NaN for every epoch, not just some -- a plain mean/std
        # over folds would make the WHOLE curve NaN (invisible) instead of just excluding
        # that one fold, see presence_balanced_acc/presence_auroc's own NaN convention.
        mean, std = np.nanmean(bacc, axis=0), np.nanstd(bacc, axis=0)
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
    png_path = out_dir / f"rmse_vs_epoch_{out_tag}.png"
    plt.savefig(png_path, dpi=200)
    plt.savefig(out_dir / f"rmse_vs_epoch_{out_tag}.pdf")
    print(f"saved to {png_path} and .pdf")


if __name__ == "__main__":
    main()
