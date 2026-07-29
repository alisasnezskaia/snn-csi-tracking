"""Presence (binary) + position (2D) SNN, trained on the raw plain-text CSI
captures + MediaPipe-extracted video trajectories under data/raw_captures/
(see data/raw_capture_loader.py's docstring for the expected layout).

Ported from the working notebook (csi_tracking_clean) that got 76.7% test
presence accuracy on a first attempt -- T_WIN=64/STRIDE=32 here matches what
that notebook's MAIN run actually used (not the T_WIN=128/STRIDE=64
mentioned in an earlier, superseded config cell).

Unlike the notebook, spikes are encoded per-batch inside the training loop
rather than precomputed for the whole train/val/test split up front -- each
window is 3x1024xT_WIN (~1.5MB at T_WIN=128, less at 64), and materializing
a same-size spike tensor for an entire split risks the memmap/OOM failure
mode data/mat_loader.py and data/dataset.py's CSIDataset already document
elsewhere in this repo.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from snn_csi_tracking.data.dataset import build_datasets
from snn_csi_tracking.data.presence_position_dataset import load_or_build_dataset
from snn_csi_tracking.models.presence_position_snn import SNNPresencePosition, to_spikes

REPO_ROOT = Path(__file__).parent.parent.parent.parent
RAW_ROOT = REPO_ROOT / "data" / "raw_captures"
TRAJECTORY_CACHE_DIR = REPO_ROOT / "data" / "processed" / "presence_position_trajectories"
CACHE_DIR = REPO_ROOT / "data" / "processed"

RATE_MS = 50
T_WIN = 64
STRIDE = 32
USE_PHASE = True  # cross-antenna phase-difference channels alongside amplitude,
                   # see data/presence_position_dataset.compute_features

MODEL_OUT = REPO_ROOT / "results" / "models" / f"presence_position_snn_{'ampphase' if USE_PHASE else 'amp'}_50ms.pt"

DELTA_THRESHOLD = 0.3
HIDDEN_1 = 128
HIDDEN_2 = 32

LEARNING_RATE = 5e-4
BATCH_SIZE = 32
# 20 epochs (the notebook's original choice) turned out not to be enough --
# the per-frame model's train loss and val accuracy were both still moving
# at epoch 20 (see conversation), not plateaued. 60 gives real headroom to
# check for convergence instead of guessing.
NUM_EPOCHS = 60
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_dataloaders(X, y, present, groups, batch_size):
    train_ds, val_ds, test_ds = build_datasets(X, y, groups, label_dtype=torch.float32, extra=present)
    print(f"split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} windows")
    print(
        f"presence rate -- train: {train_ds.extra.mean():.2f}, "
        f"val: {val_ds.extra.mean():.2f}, test: {test_ds.extra.mean():.2f}"
    )
    loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), persistent_workers=NUM_WORKERS > 0)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        train_ds,
    )


def train_one_epoch(model, loader, optimizer, bce_loss, mse_loss) -> float:
    model.train()
    total_loss, total = 0.0, 0
    for windows, xy, present in loader:
        windows, xy, present = windows.to(DEVICE), xy.to(DEVICE), present.to(DEVICE)
        optimizer.zero_grad()
        spikes = to_spikes(windows, threshold=DELTA_THRESHOLD)
        pres_pred, pos_pred = model(spikes)
        loss_presence = bce_loss(pres_pred, present)
        pos_err = mse_loss(pos_pred, xy).mean(dim=1)
        loss_position = (pos_err * present).sum() / (present.sum() + 1e-8)
        loss = loss_presence + loss_position
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * present.size(0)
        total += present.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, bce_loss) -> tuple[float, float, float]:
    """Returns (loss, presence_acc, position_error)."""
    model.eval()
    all_pres_pred, all_pos_pred, all_present, all_xy = [], [], [], []
    for windows, xy, present in loader:
        windows, xy, present = windows.to(DEVICE), xy.to(DEVICE), present.to(DEVICE)
        spikes = to_spikes(windows, threshold=DELTA_THRESHOLD)
        pres_pred, pos_pred = model(spikes)
        all_pres_pred.append(pres_pred)
        all_pos_pred.append(pos_pred)
        all_present.append(present)
        all_xy.append(xy)
    pres_pred = torch.cat(all_pres_pred)
    pos_pred = torch.cat(all_pos_pred)
    present = torch.cat(all_present)
    xy = torch.cat(all_xy)

    presence_acc = ((torch.sigmoid(pres_pred) > 0.5).float() == present).float().mean().item()
    pos_err = (torch.norm(pos_pred - xy, dim=1) * present).sum() / (present.sum() + 1e-8)
    loss = bce_loss(pres_pred, present).item() + pos_err.item()
    return loss, presence_acc, pos_err.item()


def main():
    if not RAW_ROOT.exists():
        sys.exit(
            f"No raw captures found at {RAW_ROOT}.\n"
            "This pipeline needs the 50 plain-text CSI capture files + paired videos "
            "(see data/raw_capture_loader.py's docstring for the expected layout: "
            "data/raw_captures/{condition}_{activity}/capture{n}.mat + videos/segment{n}.mp4).\n"
            "Sync them in from Drive first, then re-run."
        )

    t_start = time.time()
    print(f"Loading {RATE_MS}ms captures, T_WIN={T_WIN}, STRIDE={STRIDE}, USE_PHASE={USE_PHASE}...")
    X, y, present, groups, activity_codes = load_or_build_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, CACHE_DIR, rate_ms=RATE_MS, t_win=T_WIN, stride=STRIDE, use_phase=USE_PHASE
    )
    print(f"X={X.shape}, {len(set(groups))} trials ({time.time()-t_start:.1f}s)")

    train_loader, val_loader, test_loader, train_ds = make_dataloaders(X, y, present, groups, BATCH_SIZE)

    pos_weight = torch.tensor([(1 - train_ds.extra.mean().item()) / train_ds.extra.mean().item()]).to(DEVICE)
    print(f"pos_weight = {pos_weight.item():.3f}")

    model = SNNPresencePosition(in_features=X.shape[1] * X.shape[2], h1=HIDDEN_1, h2=HIDDEN_2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse_loss = nn.MSELoss(reduction="none")

    best_val_loss, best_state, best_epoch = float("inf"), None, None

    print(f"\nTraining on {DEVICE} for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, bce_loss, mse_loss)
        val_loss, val_presence_acc, val_pos_err = evaluate(model, val_loader, bce_loss)
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            marker = "  <- best so far"
        print(
            f"epoch {epoch:2d}/{NUM_EPOCHS}  train_loss={train_loss:.5f}  "
            f"val_presence_acc={val_presence_acc:.3f}  val_pos_err={val_pos_err:.4f}  "
            f"({time.time()-t0:.1f}s){marker}"
        )

    print(f"\nrestoring best checkpoint from epoch {best_epoch} (val_loss={best_val_loss:.5f})")
    model.load_state_dict(best_state)

    _test_loss, test_presence_acc, test_pos_err = evaluate(model, test_loader, bce_loss)
    print(f"\nFINAL TEST: presence_acc={test_presence_acc:.3f}, position_error={test_pos_err:.4f}")
    print(f"total wall time: {time.time()-t_start:.1f}s")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_OUT)
    print(f"saved best checkpoint to {MODEL_OUT}")


if __name__ == "__main__":
    main()
