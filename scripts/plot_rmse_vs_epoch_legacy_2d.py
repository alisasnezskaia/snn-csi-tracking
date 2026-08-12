"""Byte-faithful reconstruction of the 2D RMSE-vs-epoch script exactly as it
existed on 2026-07-24 (the run that produced this repo's original
results/figures/rmse_vs_epoch.png and results/rmse_vs_epoch_history.npz) --
recovered from that day's Claude Code session transcript
(~/.claude/projects/-home-newuser-Downloads-snn-csi-tracking/
6518a6b4-eea1-4edc-9b79-3d52b8dff5f7.jsonl), since no git history predates
2026-07-29 and this exact code no longer exists anywhere else.

Why a separate script instead of reverting train_presence_position_conv.py:
that shared module's train_one_epoch/evaluate were changed on 2026-07-29 to
require mu/sigma (position_norm_stats, added for the 3D pinhole meters-scale
fix -- real-meters position error was dwarfing the presence BCE loss and
starving presence learning through the shared trunk). Every 2D/3D/3dz script
written since (including today's --no-pinhole work) depends on that current
signature, so reverting it in place would break all of them. This script
instead locally reimplements the two functions exactly as they were on
2026-07-24 -- no position normalization at all, raw [0,1] position target
directly in the MSE loss (which is why it never needed normalizing: 2D (x,y)
is already MediaPipe's own [0,1] convention, so the scale-mismatch problem
that motivated position_norm_stats never applied here).

collect_predictions and rmse_and_presence_acc are reused as-is from the
current shared module -- both are unchanged/backward-compatible (mu/sigma
are optional kwargs there, defaulting to exactly this raw-scale behavior).

Run:
    .venv/bin/python scripts/plot_rmse_vs_epoch_legacy_2d.py
Output (distinct filenames -- deliberately does NOT overwrite the original
2026-07-24 artifacts, so they stay available as a direct comparison target):
    results/figures/rmse_vs_epoch_legacy.png (and .pdf)
    results/rmse_vs_epoch_history_legacy.npz
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
    collect_predictions, rmse_and_presence_acc, session_key,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONDITIONS = ("NLoS", "PLoS")
ROTATION = ["EN-S", "EN-W", "L", "S"]  # E excluded -- always kept in train

MODELS = ["snn", "ann", "lstm"]
LABELS = {"snn": "SNN (best, cross\\_coherence)", "ann": "ANN", "lstm": "LSTM"}
COLORS = {"snn": "#2563eb", "ann": "#dc2626", "lstm": "#eab308"}  # final color as of the 2026-07-24 run


def legacy_train_one_epoch(model, loader, optimizer, bce_loss, smoothness_weight: float) -> float:
    """== train_presence_position_conv.train_one_epoch as of 2026-07-24 (pre-position_norm_stats)."""
    model.train()
    total_loss, total = 0.0, 0
    for windows, pos_seq, pres_seq in loader:
        windows = windows.to(DEVICE)
        pos_seq = pos_seq.to(DEVICE).permute(1, 0, 2)  # (T, batch, out_dim)
        pres_seq = pres_seq.to(DEVICE).permute(1, 0)  # (T, batch)
        optimizer.zero_grad()
        pres_pred, pos_pred = model(windows)
        loss_presence = bce_loss(pres_pred, pres_seq).mean()
        pos_err = ((pos_pred - pos_seq) ** 2).sum(dim=-1)
        loss_position = (pos_err * pres_seq).sum() / (pres_seq.sum() + 1e-8)
        loss = loss_presence + loss_position
        if smoothness_weight > 0:
            both_present = pres_seq[1:] * pres_seq[:-1]
            jump = ((pos_pred[1:] - pos_pred[:-1]) ** 2).sum(dim=-1)
            loss_smoothness = (jump * both_present).sum() / (both_present.sum() + 1e-8)
            loss = loss + smoothness_weight * loss_smoothness
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pres_seq.shape[1]
        total += pres_seq.shape[1]
    return total_loss / total


@torch.no_grad()
def legacy_evaluate(model, loader, bce_loss) -> tuple[float, float, float]:
    """== train_presence_position_conv.evaluate as of 2026-07-24 (pre-position_norm_stats).
    Returns (loss, presence_acc, rmse)."""
    pres_pred, pos_pred, pres, pos = collect_predictions(model, loader)
    rmse, presence_acc = rmse_and_presence_acc(pres_pred, pos_pred, pres, pos)
    sq_err = ((pos_pred - pos) ** 2).sum(dim=-1)
    mse = (sq_err * pres).sum() / (pres.sum() + 1e-8)
    loss = bce_loss(pres_pred, pres).mean().item() + mse.item()
    return loss, presence_acc, rmse


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


def run_fold(model_type: str, X, pos, present, groups, test_activity: str, val_activity: str) -> np.ndarray:
    """Trains one fold for NUM_EPOCHS, returns the per-epoch val_rmse array (NUM_EPOCHS,)."""
    train_loader, val_loader, _test_loader, train_ds = make_fold_loaders(X, pos, present, groups, test_activity, val_activity)
    train_presence_rate = train_ds.extra.float().mean().item()
    pos_weight = torch.tensor([(1 - train_presence_rate) / train_presence_rate]).to(DEVICE)

    model = build_model(model_type, X.shape[1], X.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    rmse_history = np.zeros(NUM_EPOCHS)
    for epoch in range(NUM_EPOCHS):
        legacy_train_one_epoch(model, train_loader, optimizer, bce_loss, SMOOTHNESS_WEIGHT)
        _val_loss, _val_presence_acc, val_rmse = legacy_evaluate(model, val_loader, bce_loss)
        rmse_history[epoch] = val_rmse
    return rmse_history


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
    t_start = time.time()
    for model_type in MODELS:
        for i, test_activity in enumerate(ROTATION):
            val_activity = ROTATION[(i + 1) % len(ROTATION)]
            t0 = time.time()
            history[model_type][i] = run_fold(model_type, X, pos, present, groups, test_activity, val_activity)
            print(f"{model_type:4s} fold {i+1}/{len(ROTATION)} (test={test_activity}, val={val_activity}): "
                  f"final val_rmse={history[model_type][i, -1]:.4f}  ({time.time()-t0:.1f}s)")
    print(f"total wall time: {time.time()-t_start:.1f}s")

    out_dir = REPO_ROOT / "results"
    np.savez(out_dir / "rmse_vs_epoch_history_legacy.npz", epochs=np.arange(1, NUM_EPOCHS + 1),
             **{f"{m}_rmse": history[m] for m in MODELS})
    print(f"saved raw per-fold history to {out_dir / 'rmse_vs_epoch_history_legacy.npz'}")

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
    ax.set_ylabel("Validation RMSE (normalized position)")
    ax.set_xlim(1, NUM_EPOCHS)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, which="major")
    ax.legend(frameon=False)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "rmse_vs_epoch_legacy.png", dpi=200)
    plt.savefig(out_dir / "rmse_vs_epoch_legacy.pdf")
    print(f"saved to {out_dir / 'rmse_vs_epoch_legacy.png'} and .pdf")


if __name__ == "__main__":
    main()
