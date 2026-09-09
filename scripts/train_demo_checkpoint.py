"""Train ONE full SNNPresencePositionConvPerFrame checkpoint using the
current best full-pipeline config (feature=cross_coherence -- see
train_presence_position_cv.py for the honest leave-activity-out numbers;
earlier "clean" cross_coherence runs were confounded by a CFAR/adaptive-
threshold bug, so no specific number is cited here), holding BOTH L and
EN-W out of train entirely -- so the spy-overlay video rendered from this
checkpoint for an L trial and an EN-W trial is a genuine "never seen this
activity" prediction, not a training-set replay.

Not a CV script (train_presence_position_cv.py already covers the honest
per-fold metrics) -- this exists purely to produce ONE deployable
checkpoint for scripts/render_spy_overlay_conv.py's video sanity check.

Run:
    .venv/bin/python scripts/train_demo_checkpoint.py
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

from snn_csi_tracking.data.dataset import CSIDataset
from snn_csi_tracking.data.presence_position_dataset import load_or_build_perframe_dataset
from snn_csi_tracking.models.presence_position_snn import SNNPresencePositionConvPerFrame
from snn_csi_tracking.training.train_presence_position_conv import (
    AMPLITUDE_NORM, BATCH_SIZE, CACHE_DIR, CONV_CHANNELS, DELTA_THRESHOLD, DEPTH_CACHE_DIR,
    HIDDEN_1, HIDDEN_2, KERNEL_SIZE, LEARNING_RATE, NUM_EPOCHS, NUM_WORKERS, RATE_MS, RAW_ROOT,
    STRIDE, T_WIN, TRAJECTORY_CACHE_DIR, USE_EMPTY_BASELINE, USE_MOTION_MAGNITUDE, USE_PHASE,
    calibrate_presence_threshold, evaluate, session_key, train_one_epoch,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_OUT = REPO_ROOT / "results" / "models" / "presence_position_snn_cross_coherence_demo.pt"

TEST_ACTIVITIES = {"EN-W", "L"}   # held fully out of train -- these are the 2 demo videos
VAL_ACTIVITIES = {"EN-S"}         # held out too, only for early-stopping + threshold calibration --
                                   # NOT "S": S is present in 100% of its frames (never empty), so a
                                   # threshold sweep against it has no negative examples at all --
                                   # presence_balanced_acc returns nan for every threshold, and
                                   # calibrate_presence_threshold silently falls back to an
                                   # uncalibrated 0.5 (confirmed: this broke the first attempt --
                                   # balanced_acc=-1.000, the function's un-overwritten sentinel).
                                   # EN-S starts empty then transitions to present, giving both classes.


def main():
    print(f"Loading dataset (feature=cross_coherence, holding out {TEST_ACTIVITIES} from train)...")
    X, pos, present, groups, activity_codes = load_or_build_perframe_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, DEPTH_CACHE_DIR, CACHE_DIR,
        rate_ms=RATE_MS, t_win=T_WIN, stride=STRIDE, use_3d=False, use_phase=USE_PHASE,
        use_empty_baseline=USE_EMPTY_BASELINE, denoise=None, use_motion_magnitude=USE_MOTION_MAGNITUDE,
        amplitude_norm=AMPLITUDE_NORM, use_relative_motion=False, use_cross_coherence=True,
    )
    print(f"X={X.shape}, {len(set(groups))} trials")

    sessions = np.array([session_key(g) for g in groups])
    activity_of_session = {s: s.split("_", 1)[1] for s in set(sessions)}
    bucket = np.array([
        "test" if activity_of_session[s] in TEST_ACTIVITIES
        else "val" if activity_of_session[s] in VAL_ACTIVITIES
        else "train"
        for s in sessions
    ])
    train_idx, val_idx, test_idx = np.where(bucket == "train")[0], np.where(bucket == "val")[0], np.where(bucket == "test")[0]
    train_ds = CSIDataset(X, pos, train_idx, label_dtype=torch.float32, extra=present)
    val_ds = CSIDataset(X, pos, val_idx, label_dtype=torch.float32, extra=present)
    print(f"train sessions: {sorted(set(sessions[train_idx]))}")
    print(f"val sessions:   {sorted(set(sessions[val_idx]))}")
    print(f"test sessions (not used in training, demo-only): {sorted(set(sessions[test_idx]))}")

    loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    train_presence_rate = train_ds.extra.float().mean().item()
    pos_weight = torch.tensor([(1 - train_presence_rate) / train_presence_rate]).to(DEVICE)

    model = SNNPresencePositionConvPerFrame(
        num_channels=X.shape[1], num_subcarriers=X.shape[2], conv_channels=CONV_CHANNELS,
        h1=HIDDEN_1, h2=HIDDEN_2, out_dim=2, kernel_size=KERNEL_SIZE, delta_threshold=DELTA_THRESHOLD,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    best_val_loss, best_state, best_epoch = float("inf"), None, None
    t_start = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, bce_loss, smoothness_weight=1.0)
        val_loss, val_presence_acc, val_rmse = evaluate(model, val_loader, bce_loss)
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = "  <- best so far"
        print(f"epoch {epoch:2d}/{NUM_EPOCHS}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
              f"val_presence_acc={val_presence_acc:.3f}  val_rmse={val_rmse:.4f}  ({time.time()-t0:.1f}s){marker}")

    print(f"\nrestoring best checkpoint from epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    model.load_state_dict(best_state)
    calib_threshold, calib_bacc = calibrate_presence_threshold(model, val_loader)
    print(f"calibrated presence threshold: {calib_threshold:.2f} (balanced_acc={calib_bacc:.3f})")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_OUT)
    print(f"saved checkpoint to {MODEL_OUT}")
    print(f"total wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
