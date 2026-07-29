"""Leave-activity-out CV for ImageCNNPresenceClassifier -- direct test of
"treat the window as a photo, use a real 2D CNN" (see conversation).

Window-level presence labels (not per-frame): a window counts as "present"
if >=50% of its frames are present, same convention already used by
presence_position_dataset.make_position_windows -- this model makes ONE
holistic judgment per window, not a per-frame sequence, so it needs a
matching window-level target.

Same 4-fold leave-activity-out protocol, same best-of-best preprocessing
(relative motion, energy amplitude normalization) as
train_presence_position_cv.py -- directly comparable to its summary
(presence_acc=0.627+/-0.066, AUROC=0.542+/-0.061).

Run:
    .venv/bin/python scripts/train_image_cnn_presence.py
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
from snn_csi_tracking.models.image_cnn_presence import ImageCNNPresenceClassifier
from snn_csi_tracking.training.train_presence_position_conv import (
    AMPLITUDE_NORM, BATCH_SIZE, CACHE_DIR, DEPTH_CACHE_DIR, LEARNING_RATE, NUM_EPOCHS, NUM_WORKERS,
    RATE_MS, RAW_ROOT, STRIDE, T_WIN, TRAJECTORY_CACHE_DIR, USE_EMPTY_BASELINE, USE_MOTION_MAGNITUDE,
    USE_PHASE, USE_RELATIVE_MOTION, session_key,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONDITIONS = ("NLoS", "PLoS")
ROTATION = ["EN-S", "EN-W", "L", "S"]
MIN_VALID_FRAC = 0.5


def window_labels(present_perframe: np.ndarray) -> np.ndarray:
    """present_perframe: (NumWindows, t_win) -- returns (NumWindows,) float32,
    1.0 if >=50% of the window's frames are present, matching
    make_position_windows' own convention."""
    return (present_perframe.mean(axis=1) >= MIN_VALID_FRAC).astype(np.float32)


def make_fold_loaders(X, labels, groups, test_activity: str, val_activity: str):
    sessions = np.array([session_key(g) for g in groups])
    test_sessions = {f"{c}_{test_activity}" for c in CONDITIONS}
    val_sessions = {f"{c}_{val_activity}" for c in CONDITIONS}
    bucket = np.array([
        "test" if s in test_sessions else "val" if s in val_sessions else "train" for s in sessions
    ])
    train_idx, val_idx, test_idx = np.where(bucket == "train")[0], np.where(bucket == "val")[0], np.where(bucket == "test")[0]
    train_ds = CSIDataset(X, labels, train_idx, label_dtype=torch.float32)
    val_ds = CSIDataset(X, labels, val_idx, label_dtype=torch.float32)
    test_ds = CSIDataset(X, labels, test_idx, label_dtype=torch.float32)
    loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), persistent_workers=NUM_WORKERS > 0)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs),
        train_ds,
    )


def calibrate_threshold(probs: np.ndarray, gt: np.ndarray) -> float:
    best_t, best_bacc = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (probs > t).astype(float)
        pos_mask, neg_mask = gt == 1, gt == 0
        sens = pred[pos_mask].mean() if pos_mask.any() else float("nan")
        spec = (1 - pred[neg_mask]).mean() if neg_mask.any() else float("nan")
        bacc = np.nanmean([sens, spec])
        if bacc > best_bacc:
            best_bacc, best_t = bacc, float(t)
    return best_t


@torch.no_grad()
def collect(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    for windows, labels in loader:
        windows, labels = windows.to(DEVICE), labels.to(DEVICE)
        logits = model(windows)
        all_logits.append(logits)
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def run_fold(X, labels, groups, test_activity: str, val_activity: str) -> dict:
    train_loader, val_loader, test_loader, train_ds = make_fold_loaders(X, labels, groups, test_activity, val_activity)
    train_rate = train_ds.labels.float().mean().item()
    train_rate = min(max(train_rate, 0.05), 0.95)
    pos_weight = torch.tensor([(1 - train_rate) / train_rate]).to(DEVICE)

    model = ImageCNNPresenceClassifier(num_channels=X.shape[1], num_subcarriers=X.shape[2], t_win=X.shape[3]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss, best_state = float("inf"), None
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for windows, lbl in train_loader:
            windows, lbl = windows.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()
            loss = bce_loss(model(windows), lbl)
            loss.backward()
            optimizer.step()
        val_logits, val_labels = collect(model, val_loader)
        val_loss = bce_loss(val_logits, val_labels).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    val_logits, val_labels = collect(model, val_loader)
    val_probs = torch.sigmoid(val_logits).cpu().numpy()
    threshold = calibrate_threshold(val_probs, val_labels.cpu().numpy())

    test_logits, test_labels = collect(model, test_loader)
    test_probs = torch.sigmoid(test_logits).cpu().numpy()
    test_gt = test_labels.cpu().numpy()
    presence_acc = ((test_probs > threshold).astype(float) == test_gt).mean()
    auroc = roc_auc_score(test_gt, test_probs) if len(set(test_gt)) > 1 else float("nan")
    return {"test_activity": test_activity, "presence_acc": float(presence_acc), "auroc": auroc, "threshold": threshold}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-features", action="store_true",
                         help="stack spectral_ratio + cross_coherence on top of relative_motion")
    args = parser.parse_args()

    print(f"Loading dataset (amplitude_norm={AMPLITUDE_NORM}, use_relative_motion={USE_RELATIVE_MOTION}, "
          f"all_features={args.all_features})...")
    X, pos, present, groups, activity_codes = load_or_build_perframe_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, DEPTH_CACHE_DIR, CACHE_DIR,
        rate_ms=RATE_MS, t_win=T_WIN, stride=STRIDE, use_3d=False, use_phase=USE_PHASE,
        use_empty_baseline=USE_EMPTY_BASELINE, denoise=None, use_motion_magnitude=USE_MOTION_MAGNITUDE,
        amplitude_norm=AMPLITUDE_NORM, use_relative_motion=USE_RELATIVE_MOTION,
        use_spectral_ratio=args.all_features, use_cross_coherence=args.all_features,
    )
    labels = window_labels(present)
    print(f"window-level presence rate: {labels.mean():.3f}")

    results = []
    t_start = time.time()
    for i, test_activity in enumerate(ROTATION):
        val_activity = ROTATION[(i + 1) % len(ROTATION)]
        print(f"\n=== image-CNN fold {i+1}/{len(ROTATION)}: test={test_activity}, val={val_activity} ===")
        t0 = time.time()
        result = run_fold(X, labels, groups, test_activity, val_activity)
        results.append(result)
        print(f"  presence_acc={result['presence_acc']:.3f}  auroc={result['auroc']:.3f}  "
              f"threshold={result['threshold']:.2f}  ({time.time()-t0:.1f}s)")

    accs = [r["presence_acc"] for r in results]
    aurocs = [r["auroc"] for r in results if not np.isnan(r["auroc"])]
    print(f"\n=== image-CNN cross-validation summary ({len(ROTATION)} folds) ===")
    print(f"presence_acc: {np.mean(accs):.3f} +/- {np.std(accs):.3f}   (per-fold: {[f'{a:.3f}' for a in accs]})")
    print(f"AUROC:        {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}   (per-fold: {[f'{a:.3f}' for a in aurocs]})")
    print(f"total wall time: {time.time()-t_start:.1f}s")
    print("(compare against relative-motion baseline: presence_acc=0.627+/-0.066, AUROC=0.542+/-0.061)")


if __name__ == "__main__":
    main()
