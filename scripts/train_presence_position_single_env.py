"""Same-environment, single-calibration evaluation -- the clean version of
"can this SNN detect presence and estimate position within ONE calibrated
environment", distinct from both evaluations already run:
  - chronological split (train_presence_position_conv.py's `--split-mode`
    default path): leaky, same continuous recording, different time slices
  - leave-activity-out CV (train_presence_position_cv.py): zero-shot to a
    session the model never saw at all

Here, each of the 10 sessions (5 activities x {NLoS, PLoS}) gets its OWN
dedicated model, trained and tested ONLY on that session's own captures:
    train = captures 1-3, val = capture 4, test = capture 5
No cross-session generalization is being asked of the model at all -- this
is the realistic "install a sensor in a room, calibrate it there, use it
there" deployment story, and it's the number that actually supports the
abstract's joint presence+position claim without needing to solve the
(separately characterized, harder) cross-environment generalization problem.

Reports presence_acc, AUROC, and position RMSE per session, then mean+/-std
across all 10 -- directly comparable to the leave-activity-out CV's summary
line (presence_acc=0.646+/-0.050, AUROC=0.508+/-0.007, RMSE=0.2606+/-0.0793).

Run:
    .venv/bin/python scripts/train_presence_position_single_env.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_csi_tracking.data.dataset import CSIDataset
from snn_csi_tracking.data.presence_position_dataset import load_or_build_perframe_dataset
from snn_csi_tracking.models.presence_position_snn import SNNPresencePositionConvPerFrame
from snn_csi_tracking.training.train_presence_position_conv import (
    AMPLITUDE_NORM, CACHE_DIR, CONV_CHANNELS, DELTA_THRESHOLD, DEPTH_CACHE_DIR, HIDDEN_1, HIDDEN_2,
    KERNEL_SIZE, LEARNING_RATE, NUM_EPOCHS, NUM_WORKERS, RATE_MS, RAW_ROOT, SMOOTHNESS_WEIGHT, STRIDE, T_WIN,
    TRAJECTORY_CACHE_DIR, USE_EMPTY_BASELINE, USE_MOTION_MAGNITUDE, USE_PHASE, USE_RELATIVE_MOTION,
    calibrate_presence_threshold, collect_predictions, evaluate, presence_balanced_acc, rmse_and_presence_acc,
    session_key, train_one_epoch,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONDITIONS = ("NLoS", "PLoS")
ACTIVITIES = ("E", "EN-S", "EN-W", "L", "S")
BATCH_SIZE = 16  # smaller batches -- each session only has ~180 windows total


def presence_auroc(pres_pred: torch.Tensor, pres_gt: torch.Tensor) -> float:
    probs = torch.sigmoid(pres_pred).cpu().numpy().ravel()
    gt = pres_gt.cpu().numpy().ravel()
    return roc_auc_score(gt, probs) if len(set(gt)) > 1 else float("nan")


def run_session(X, pos, present, groups, session: str) -> dict | None:
    sessions = np.array([session_key(g) for g in groups])
    session_mask = np.where(sessions == session)[0]
    capture_num = np.array([int(groups[i].rsplit("capture", 1)[-1]) for i in session_mask])

    train_idx = session_mask[np.isin(capture_num, [1, 2, 3])]
    val_idx = session_mask[capture_num == 4]
    test_idx = session_mask[capture_num == 5]
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        print(f"  skipping {session}: missing a capture bucket (train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")
        return None

    train_ds = CSIDataset(X, pos, train_idx, label_dtype=torch.float32, extra=present)
    val_ds = CSIDataset(X, pos, val_idx, label_dtype=torch.float32, extra=present)
    test_ds = CSIDataset(X, pos, test_idx, label_dtype=torch.float32, extra=present)
    loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    train_presence_rate = train_ds.extra.float().mean().item()
    train_presence_rate = min(max(train_presence_rate, 0.05), 0.95)  # avoid div-by-~0 on a near-constant session
    pos_weight = torch.tensor([(1 - train_presence_rate) / train_presence_rate]).to(DEVICE)

    model = SNNPresencePositionConvPerFrame(
        num_channels=X.shape[1], num_subcarriers=X.shape[2], conv_channels=CONV_CHANNELS,
        h1=HIDDEN_1, h2=HIDDEN_2, out_dim=2, kernel_size=KERNEL_SIZE, delta_threshold=DELTA_THRESHOLD,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    best_val_loss, best_state = float("inf"), None
    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, bce_loss, SMOOTHNESS_WEIGHT)
        val_loss, _val_presence_acc, _val_rmse = evaluate(model, val_loader, bce_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    calib_threshold, _calib_bacc = calibrate_presence_threshold(model, val_loader)
    test_pres_pred, test_pos_pred, test_pres_gt, test_pos_gt = collect_predictions(model, test_loader)
    rmse, presence_acc = rmse_and_presence_acc(test_pres_pred, test_pos_pred, test_pres_gt, test_pos_gt, threshold=calib_threshold)
    auroc = presence_auroc(test_pres_pred, test_pres_gt)
    return {"session": session, "presence_acc": presence_acc, "auroc": auroc, "rmse": rmse, "threshold": calib_threshold}


def main():
    print(f"Loading dataset (amplitude_norm={AMPLITUDE_NORM})...")
    X, pos, present, groups, activity_codes = load_or_build_perframe_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, DEPTH_CACHE_DIR, CACHE_DIR,
        rate_ms=RATE_MS, t_win=T_WIN, stride=STRIDE, use_3d=False, use_phase=USE_PHASE,
        use_empty_baseline=USE_EMPTY_BASELINE, denoise=None, use_motion_magnitude=USE_MOTION_MAGNITUDE,
        amplitude_norm=AMPLITUDE_NORM, use_relative_motion=USE_RELATIVE_MOTION,
    )
    X = np.asarray(X)

    results = []
    t_start = time.time()
    for condition in CONDITIONS:
        for activity in ACTIVITIES:
            session = f"{condition}_{activity}"
            t0 = time.time()
            result = run_session(X, pos, present, groups, session)
            if result is None:
                continue
            results.append(result)
            print(f"{session}: presence_acc={result['presence_acc']:.3f}  auroc={result['auroc']:.3f}  "
                  f"rmse={result['rmse']:.4f}  threshold={result['threshold']:.2f}  ({time.time()-t0:.1f}s)")

    accs = [r["presence_acc"] for r in results]
    aurocs = [r["auroc"] for r in results if not np.isnan(r["auroc"])]
    rmses = [r["rmse"] for r in results]
    print(f"\n=== same-environment (single-calibration) summary ({len(results)} sessions) ===")
    print(f"presence_acc: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"AUROC:        {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}  (n={len(aurocs)}, excludes all-one-class sessions)")
    print(f"RMSE:         {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
    print(f"total wall time: {time.time()-t_start:.1f}s")
    print("(compare against leave-activity-out CV: presence_acc=0.646+/-0.050, AUROC=0.508+/-0.007, RMSE=0.2606+/-0.0793)")

    np.savez(REPO_ROOT / "results" / "single_env_results.npz",
             sessions=[r["session"] for r in results], presence_acc=accs, rmse=rmses,
             auroc=[r["auroc"] for r in results])
    print(f"saved per-session results to results/single_env_results.npz")


if __name__ == "__main__":
    main()
