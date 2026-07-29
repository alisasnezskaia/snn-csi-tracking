"""Same leave-activity-out CV protocol and model config as
train_presence_position_cv.py, but using the on-the-fly windowing path
(presence_position_dataset.build_capture_cache + SlidingWindowCaptureDataset)
instead of a pre-materialized window array -- see that module's docstrings
for why: a small stride like 16 roughly doubles the window count over the
default stride=32, and this run's feature set (cross_coherence +
baseline_deviation) is heavier than the others (baseline_deviation adds 3
full-resolution channels, not a broadcast scalar), which is exactly the
combination that kept getting killed (exit 137) building one big
pre-materialized array (see conversation). This instead caches each
capture's full feature array ONCE (~40MB/capture) and slices out windows
on demand, so memory no longer scales with how small the stride is.

Defaults to feature=cross_coherence + baseline_deviation (the current best
full-pipeline config, now with the num_static_channels architecture fix so
the static feature isn't delta-encoded away -- see presence_position_snn.
SNNPresencePositionConvPerFrame's docstring) and --stride 16.

Run:
    .venv/bin/python scripts/train_presence_position_cv_dense.py --stride 16
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

from snn_csi_tracking.data.presence_position_dataset import build_capture_cache, SlidingWindowCaptureDataset
from snn_csi_tracking.data.raw_capture_loader import NUM_ANTENNAS
from snn_csi_tracking.models.presence_position_snn import SNNPresencePositionConvPerFrame
from snn_csi_tracking.training.train_presence_position_conv import (
    AMPLITUDE_NORM, BATCH_SIZE, CACHE_DIR, CONV_CHANNELS, DELTA_THRESHOLD, HIDDEN_1, HIDDEN_2,
    KERNEL_SIZE, LEARNING_RATE, NUM_EPOCHS, NUM_WORKERS, RAW_ROOT, SMOOTHNESS_WEIGHT, T_WIN,
    TRAJECTORY_CACHE_DIR, USE_EMPTY_BASELINE, USE_MOTION_MAGNITUDE, USE_PHASE,
    calibrate_presence_threshold, collect_predictions, evaluate, rmse_and_presence_acc, train_one_epoch,
)
from train_presence_position_cv import presence_auroc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONDITIONS = ("NLoS", "PLoS")
ROTATION = ["EN-S", "EN-W", "L", "S"]  # E excluded -- always kept in train, see train_presence_position_cv.py


def session_of(trial_key: str) -> str:
    return trial_key.rsplit("_", 1)[0]  # "{condition}_{activity}_captureN" -> "{condition}_{activity}"


def run_fold(capture_cache: dict, test_activity: str, val_activity: str, t_win: int, stride: int,
             num_static_channels: int) -> dict:
    test_sessions = {f"{c}_{test_activity}" for c in CONDITIONS}
    val_sessions = {f"{c}_{val_activity}" for c in CONDITIONS}
    train_keys, val_keys, test_keys = [], [], []
    for tk in capture_cache:
        s = session_of(tk)
        (test_keys if s in test_sessions else val_keys if s in val_sessions else train_keys).append(tk)

    train_ds = SlidingWindowCaptureDataset(capture_cache, train_keys, t_win, stride)
    val_ds = SlidingWindowCaptureDataset(capture_cache, val_keys, t_win, stride)
    test_ds = SlidingWindowCaptureDataset(capture_cache, test_keys, t_win, stride)
    print(f"  windows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    train_presence_rate = np.mean([capture_cache[tk]["present"].mean() for tk in train_keys])
    train_presence_rate = min(max(train_presence_rate, 0.05), 0.95)
    pos_weight = torch.tensor([(1 - train_presence_rate) / train_presence_rate]).to(DEVICE)

    num_channels = capture_cache[train_keys[0]]["feat"].shape[0]
    num_subcarriers = capture_cache[train_keys[0]]["feat"].shape[1]
    model = SNNPresencePositionConvPerFrame(
        num_channels=num_channels, num_subcarriers=num_subcarriers, conv_channels=CONV_CHANNELS,
        h1=HIDDEN_1, h2=HIDDEN_2, out_dim=2, kernel_size=KERNEL_SIZE, delta_threshold=DELTA_THRESHOLD,
        num_static_channels=num_static_channels,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    best_val_loss, best_state = float("inf"), None
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_one_epoch(model, train_loader, optimizer, bce_loss, SMOOTHNESS_WEIGHT)
        val_loss, _val_presence_acc, _val_rmse = evaluate(model, val_loader, bce_loss)
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " <- best"
        print(f"    epoch {epoch}/{NUM_EPOCHS}  val_loss={val_loss:.4f}  ({time.time()-t0:.1f}s){marker}")
    model.load_state_dict(best_state)

    calib_threshold, _calib_bacc = calibrate_presence_threshold(model, val_loader)
    test_pres_pred, test_pos_pred, test_pres_gt, test_pos_gt = collect_predictions(model, test_loader)
    rmse, presence_acc = rmse_and_presence_acc(test_pres_pred, test_pos_pred, test_pres_gt, test_pos_gt, threshold=calib_threshold)
    auroc = presence_auroc(test_pres_pred, test_pres_gt)
    return {"test_activity": test_activity, "val_activity": val_activity, "calib_threshold": calib_threshold,
            "presence_acc": presence_acc, "auroc": auroc, "rmse": rmse}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--no-baseline-deviation", action="store_true",
                         help="drop the static baseline-deviation channel, testing plain cross_coherence "
                              "at this stride instead")
    args = parser.parse_args()
    use_baseline_deviation = not args.no_baseline_deviation

    print(f"Building/loading per-capture cache (amplitude_norm={AMPLITUDE_NORM}, feature=cross_coherence, "
          f"baseline_deviation={use_baseline_deviation})...")
    capture_cache = build_capture_cache(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, CACHE_DIR, use_phase=USE_PHASE, use_empty_baseline=USE_EMPTY_BASELINE,
        denoise=None, use_motion_magnitude=USE_MOTION_MAGNITUDE, amplitude_norm=AMPLITUDE_NORM,
        use_relative_motion=False, use_cross_coherence=True, use_baseline_deviation=use_baseline_deviation,
    )
    print(f"{len(capture_cache)} captures cached, stride={args.stride}, t_win={T_WIN}")

    num_static_channels = NUM_ANTENNAS if use_baseline_deviation else 0
    results = []
    t_start = time.time()
    for i, test_activity in enumerate(ROTATION):
        val_activity = ROTATION[(i + 1) % len(ROTATION)]
        print(f"\n=== dense fold {i+1}/{len(ROTATION)}: test={test_activity}, val={val_activity} ===")
        t0 = time.time()
        result = run_fold(capture_cache, test_activity, val_activity, T_WIN, args.stride, num_static_channels)
        results.append(result)
        print(f"  presence_acc={result['presence_acc']:.3f}  auroc={result['auroc']:.3f}  "
              f"rmse={result['rmse']:.4f}  threshold={result['calib_threshold']:.2f}  ({time.time()-t0:.1f}s)")

    accs = [r["presence_acc"] for r in results]
    aurocs = [r["auroc"] for r in results if not np.isnan(r["auroc"])]
    rmses = [r["rmse"] for r in results]
    print(f"\n=== dense (stride={args.stride}) SNN cross-validation summary ({len(ROTATION)} folds) ===")
    print(f"presence_acc: {np.mean(accs):.3f} +/- {np.std(accs):.3f}   (per-fold: {[f'{a:.3f}' for a in accs]})")
    print(f"AUROC:        {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}   (per-fold: {[f'{a:.3f}' for a in aurocs]})")
    print(f"RMSE:         {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}   (per-fold: {[f'{r:.4f}' for r in rmses]})")
    print(f"total wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
