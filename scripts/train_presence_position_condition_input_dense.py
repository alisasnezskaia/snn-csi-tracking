"""Same condition-as-input architecture as
train_presence_position_condition_input.py (LoS/NLoS fed in as a known
embedding, NOT inferred as an output -- see that script's docstring for why
the auxiliary-output version underperformed), but with a dense sliding
window (stride=1 by default) using the on-the-fly windowing path
(presence_position_dataset.build_capture_cache + SlidingWindowCaptureDataset)
-- see train_presence_position_los_dense.py's docstring for why on-the-fly
windowing is needed at small strides (108GB vs 48GB free disk otherwise).

This is the combination not yet tested (see conversation): dense sampling
was previously only tried with the LoS-as-AUXILIARY-OUTPUT architecture
(train_presence_position_los_dense.py), not with condition fed in as an
input using the near-100%-accurate static classifier
(scripts/train_los_classifier.py). NUM_EPOCHS is deliberately much lower
than the stride=32 scripts' 60 for the same reason as the LoS dense
script: ~31x more windows per trial means ~31x more gradient updates per
epoch.

Run:
    .venv/bin/python scripts/train_presence_position_condition_input_dense.py --stride 1 --epochs 3
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
from snn_csi_tracking.models.presence_position_snn import SNNPresencePositionConvPerFrame
from snn_csi_tracking.training.train_presence_position_conv import (
    AMPLITUDE_NORM, BATCH_SIZE, CACHE_DIR, CONV_CHANNELS, DELTA_THRESHOLD, HIDDEN_1, HIDDEN_2,
    KERNEL_SIZE, LEARNING_RATE, NUM_WORKERS, RAW_ROOT, T_WIN, TRAJECTORY_CACHE_DIR, USE_EMPTY_BASELINE,
    USE_MOTION_MAGNITUDE, USE_PHASE, USE_RELATIVE_MOTION,
)
from train_presence_position_condition_input import (
    CONDITION_EMBED_DIM, ROTATION, calibrate_threshold, collect_predictions_cond, presence_auroc,
    train_one_epoch_cond,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONDITIONS = ("NLoS", "PLoS")


def session_of(trial_key: str) -> str:
    return trial_key.rsplit("_", 1)[0]


def run_fold(capture_cache: dict, test_activity: str, val_activity: str, t_win: int, stride: int, num_epochs: int) -> dict:
    test_sessions = {f"{c}_{test_activity}" for c in CONDITIONS}
    val_sessions = {f"{c}_{val_activity}" for c in CONDITIONS}
    train_keys, val_keys, test_keys = [], [], []
    for tk in capture_cache:
        s = session_of(tk)
        (test_keys if s in test_sessions else val_keys if s in val_sessions else train_keys).append(tk)

    condition_value = {tk: (1.0 if capture_cache[tk]["condition"] == "PLoS" else 0.0) for tk in capture_cache}
    train_ds = SlidingWindowCaptureDataset(capture_cache, train_keys, t_win, stride, condition_value)
    val_ds = SlidingWindowCaptureDataset(capture_cache, val_keys, t_win, stride, condition_value)
    test_ds = SlidingWindowCaptureDataset(capture_cache, test_keys, t_win, stride, condition_value)
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
        condition_embed_dim=CONDITION_EMBED_DIM,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    best_val_loss, best_state = float("inf"), None
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_one_epoch_cond(model, train_loader, optimizer, bce_loss)
        pres_pred, pos_pred, pres_gt, pos_gt = collect_predictions_cond(model, val_loader)
        loss_presence = bce_loss(pres_pred, pres_gt).mean()
        sq_err = ((pos_pred - pos_gt) ** 2).sum(dim=-1)
        mse = (sq_err * pres_gt).sum() / (pres_gt.sum() + 1e-8)
        val_loss = (loss_presence + mse).item()
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " <- best"
        print(f"    epoch {epoch}/{num_epochs}  val_loss={val_loss:.4f}  ({time.time()-t0:.1f}s){marker}")
    model.load_state_dict(best_state)

    val_pres_pred, _val_pos_pred, val_pres_gt, _val_pos_gt = collect_predictions_cond(model, val_loader)
    threshold = calibrate_threshold(val_pres_pred, val_pres_gt)

    pres_pred, pos_pred, pres_gt, pos_gt = collect_predictions_cond(model, test_loader)
    probs = torch.sigmoid(pres_pred)
    presence_acc = ((probs > threshold).float() == pres_gt).float().mean().item()
    auroc = presence_auroc(pres_pred, pres_gt)
    sq_err = ((pos_pred - pos_gt) ** 2).sum(dim=-1)
    rmse = torch.sqrt((sq_err * pres_gt).sum() / (pres_gt.sum() + 1e-8)).item()

    return {"test_activity": test_activity, "presence_acc": presence_acc, "auroc": auroc, "rmse": rmse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    print(f"Building/loading per-capture cache (amplitude_norm={AMPLITUDE_NORM}, use_relative_motion={USE_RELATIVE_MOTION})...")
    capture_cache = build_capture_cache(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, CACHE_DIR, use_phase=USE_PHASE, use_empty_baseline=USE_EMPTY_BASELINE,
        denoise=None, use_motion_magnitude=USE_MOTION_MAGNITUDE, amplitude_norm=AMPLITUDE_NORM,
        use_relative_motion=USE_RELATIVE_MOTION,
    )
    print(f"{len(capture_cache)} captures cached, stride={args.stride}, t_win={T_WIN}, epochs={args.epochs}")

    results = []
    t_start = time.time()
    for i, test_activity in enumerate(ROTATION):
        val_activity = ROTATION[(i + 1) % len(ROTATION)]
        print(f"\n=== dense condition-input fold {i+1}/{len(ROTATION)}: test={test_activity}, val={val_activity} ===")
        t0 = time.time()
        result = run_fold(capture_cache, test_activity, val_activity, T_WIN, args.stride, args.epochs)
        results.append(result)
        print(f"  presence_acc={result['presence_acc']:.3f}  auroc={result['auroc']:.3f}  "
              f"rmse={result['rmse']:.4f}  ({time.time()-t0:.1f}s)")

    accs = [r["presence_acc"] for r in results]
    aurocs = [r["auroc"] for r in results if not np.isnan(r["auroc"])]
    rmses = [r["rmse"] for r in results]
    print(f"\n=== dense (stride={args.stride}) condition-input CV summary ({len(ROTATION)} folds) ===")
    print(f"presence_acc: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"AUROC:        {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}")
    print(f"RMSE:         {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
    print(f"total wall time: {time.time()-t_start:.1f}s")
    print("(compare against sparse condition-input: presence_acc=0.605+/-0.143, AUROC=0.534+/-0.023, RMSE=0.2848+/-0.0780)")
    print("(compare against dense LoS-as-output: fold aurocs 0.641, 0.531, 0.446, ...)")


if __name__ == "__main__":
    main()
