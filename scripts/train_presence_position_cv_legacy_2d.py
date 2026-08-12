"""Per-fold presence_acc/AUROC/RMSE table (same protocol as
train_presence_position_cv.py) but using the LEGACY (2026-07-24,
pre-position_norm_stats) training loop from plot_rmse_vs_epoch_legacy_2d.py --
i.e. raw [0,1] position MSE, no per-fold min-max/z-score normalization.

Purpose: train_presence_position_cv.py (current code, uses position_norm_stats)
has been showing presence collapsing to chance (AUROC~0.5) on some folds in
2D and 3D runs today. This script answers the direct question "does that
collapse also happen under the exact code that produced this repo's original
good 2D results" -- same leave-activity-out folds, same checkpoint-selection-
by-best-val-balanced-accuracy logic as train_presence_position_cv.py, only
the normalization is swapped out for the recovered 2026-07-24 version.

Run:
    .venv/bin/python scripts/train_presence_position_cv_legacy_2d.py --model snn
    .venv/bin/python scripts/train_presence_position_cv_legacy_2d.py --model ann
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
from snn_csi_tracking.models.presence_position_snn import SNNPresencePositionConvPerFrame
from snn_csi_tracking.training.train_presence_position_conv import (
    AMPLITUDE_NORM, BATCH_SIZE, CACHE_DIR, CONV_CHANNELS, DELTA_THRESHOLD, DEPTH_CACHE_DIR, HIDDEN_1,
    HIDDEN_2, KERNEL_SIZE, LEARNING_RATE, NUM_EPOCHS, NUM_WORKERS, RATE_MS, RAW_ROOT, SMOOTHNESS_WEIGHT,
    STRIDE, T_WIN, TRAJECTORY_CACHE_DIR, USE_EMPTY_BASELINE, USE_MOTION_MAGNITUDE, USE_PHASE, USE_RELATIVE_MOTION,
    calibrate_presence_threshold, collect_predictions, presence_balanced_acc, rmse_and_presence_acc, session_key,
)
from plot_rmse_vs_epoch_legacy_2d import legacy_evaluate, legacy_train_one_epoch
from train_presence_position_cv import presence_auroc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONDITIONS = ("NLoS", "PLoS")
ROTATION = ["EN-S", "EN-W", "L", "S"]  # E excluded -- always kept in train


def make_fold_loaders(X, pos, present, groups, test_activity: str, val_activity: str, batch_size: int):
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
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        train_ds,
    )


def run_fold(model_type: str, X, pos, present, groups, test_activity: str, val_activity: str) -> dict:
    train_loader, val_loader, test_loader, train_ds = make_fold_loaders(
        X, pos, present, groups, test_activity, val_activity, BATCH_SIZE
    )
    train_presence_rate = train_ds.extra.float().mean().item()
    pos_weight = torch.tensor([(1 - train_presence_rate) / train_presence_rate]).to(DEVICE)

    common = dict(num_channels=X.shape[1], num_subcarriers=X.shape[2], conv_channels=CONV_CHANNELS,
                  h1=HIDDEN_1, h2=HIDDEN_2, out_dim=2, kernel_size=KERNEL_SIZE)
    if model_type == "snn":
        model = SNNPresencePositionConvPerFrame(**common, delta_threshold=DELTA_THRESHOLD, encoder_type="perframe").to(DEVICE)
    else:
        model = ANNPresencePositionConvPerFrame(**common).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    # Same checkpoint-selection convention as train_presence_position_cv.py: best val
    # BALANCED presence accuracy (fixed threshold=0.5) across epochs.
    best_val_bacc, best_state = -1.0, None
    for epoch in range(1, NUM_EPOCHS + 1):
        legacy_train_one_epoch(model, train_loader, optimizer, bce_loss, SMOOTHNESS_WEIGHT)
        val_loss, val_presence_acc, val_rmse = legacy_evaluate(model, val_loader, bce_loss)
        val_pres_pred, _val_pos_pred, val_pres_gt, _val_pos_gt = collect_predictions(model, val_loader)
        val_bacc = presence_balanced_acc(val_pres_pred, val_pres_gt, threshold=0.5)
        print(f"    epoch {epoch}/{NUM_EPOCHS}  val_loss={val_loss:.4f}  "
              f"val_presence_acc={val_presence_acc:.3f}  val_bacc={val_bacc:.3f}  val_rmse={val_rmse:.4f}")
        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    calib_threshold, _calib_bacc = calibrate_presence_threshold(model, val_loader)
    test_pres_pred, test_pos_pred, test_pres_gt, test_pos_gt = collect_predictions(model, test_loader)
    rmse, presence_acc = rmse_and_presence_acc(test_pres_pred, test_pos_pred, test_pres_gt, test_pos_gt,
                                                threshold=calib_threshold)
    auroc = presence_auroc(test_pres_pred, test_pres_gt)
    return {"test_activity": test_activity, "val_activity": val_activity, "calib_threshold": calib_threshold,
            "presence_acc": presence_acc, "auroc": auroc, "rmse": rmse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["snn", "ann"], required=True)
    parser.add_argument("--feature", choices=["none", "relative_motion", "spectral_ratio", "cross_coherence",
                                               "spectral_and_coherence", "both"], default="cross_coherence",
                         help="matches plot_rmse_vs_epoch_legacy_2d.py's hardcoded cross_coherence by default")
    args = parser.parse_args()
    use_relative_motion = args.feature in ("relative_motion", "both")
    use_spectral_ratio = args.feature in ("spectral_ratio", "both", "spectral_and_coherence")
    use_cross_coherence = args.feature in ("cross_coherence", "spectral_and_coherence")

    print(f"Loading dataset (amplitude_norm={AMPLITUDE_NORM}, feature={args.feature}, LEGACY 2026-07-24 "
          f"training loop -- no position_norm_stats)...")
    X, pos, present, groups, activity_codes = load_or_build_perframe_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, DEPTH_CACHE_DIR, CACHE_DIR,
        rate_ms=RATE_MS, t_win=T_WIN, stride=STRIDE, use_3d=False, use_phase=USE_PHASE,
        use_empty_baseline=USE_EMPTY_BASELINE, denoise=None, use_motion_magnitude=USE_MOTION_MAGNITUDE,
        amplitude_norm=AMPLITUDE_NORM, use_relative_motion=use_relative_motion, use_spectral_ratio=use_spectral_ratio,
        use_cross_coherence=use_cross_coherence,
    )

    results = []
    t_start = time.time()
    for i, test_activity in enumerate(ROTATION):
        val_activity = ROTATION[(i + 1) % len(ROTATION)]
        print(f"\n=== fold {i+1}/{len(ROTATION)}: test={test_activity}, val={val_activity}, model={args.model} (LEGACY) ===")
        t0 = time.time()
        result = run_fold(args.model, X, pos, present, groups, test_activity, val_activity)
        results.append(result)
        print(f"  presence_acc={result['presence_acc']:.3f}  auroc={result['auroc']:.3f}  "
              f"rmse={result['rmse']:.4f}  threshold={result['calib_threshold']:.2f}  ({time.time()-t0:.1f}s)")

    accs = [r["presence_acc"] for r in results]
    aurocs = [r["auroc"] for r in results if not np.isnan(r["auroc"])]
    rmses = [r["rmse"] for r in results]
    print(f"\n=== {args.model.upper()} LEGACY cross-validation summary ({len(ROTATION)} folds) ===")
    print(f"presence_acc: {np.mean(accs):.3f} +/- {np.std(accs):.3f}   (per-fold: {[f'{a:.3f}' for a in accs]})")
    print(f"AUROC:        {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}   (per-fold: {[f'{a:.3f}' for a in aurocs]})")
    print(f"RMSE:         {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}   (per-fold: {[f'{r:.4f}' for r in rmses]})")
    print(f"total wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
