"""Per-frame (50ms-resolution) presence + position SNN, with an optional 3D
(depth) extension -- ported from the notebook's "3D extension and every
50ms frame per frame training" section.

Unlike train_presence_position.py's one-label-per-window supervision, this
keeps a per-timestep label sequence and reads out a prediction at every one
of the T_WIN internal timesteps (see models/presence_position_snn.py's
SNNPresencePositionPerFrame), giving 50ms-resolution presence/position
instead of one value per 1.6s window.

USE_3D turns (x, y) into real camera-relative (X, Y, Z) in meters: the
tracked pixel is deprojected using Depth-Anything-V2's metric-indoor depth
at that point (depth_extraction.py) and the pinhole camera model
(camera_geometry.deproject_pixel_to_camera_frame) -- not room/floor-plan
coordinates, see that function's docstring for why camera-relative is a
legitimate simplification for this fixed-camera dataset. Needs the raw
per-trial videos (data/raw_captures/*/videos/), not just the cached 2D
trajectories, since depth has to be read off actual video frames.
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
from snn_csi_tracking.data.presence_position_dataset import load_or_build_perframe_dataset
from snn_csi_tracking.models.presence_position_snn import SNNPresencePositionPerFrame, to_spikes

REPO_ROOT = Path(__file__).parent.parent.parent.parent
RAW_ROOT = REPO_ROOT / "data" / "raw_captures"
TRAJECTORY_CACHE_DIR = REPO_ROOT / "data" / "processed" / "presence_position_trajectories"
DEPTH_CACHE_DIR = REPO_ROOT / "data" / "processed" / "presence_position_depth"
CACHE_DIR = REPO_ROOT / "data" / "processed"

RATE_MS = 50
T_WIN = 64
STRIDE = 32
USE_3D = True
USE_PHASE = True  # cross-antenna phase-difference channels alongside amplitude,
                   # see data/presence_position_dataset.compute_features
OUT_DIM = 3 if USE_3D else 2

# Penalizes frame-to-frame position jumps between consecutive PRESENT frames
# only (never across a presence transition, where a real jump is expected --
# e.g. someone appearing far from the 0.5 placeholder used when absent).
# Motivated by the 3D trajectory plot: predicted position was jumping around
# every frame while ground truth moves continuously, like a real person
# actually does. 0 disables it, for an A/B comparison.
SMOOTHNESS_WEIGHT = 1.0

_dim_tag = "3d" if USE_3D else "2d"
_feat_tag = "ampphase" if USE_PHASE else "amp"
_smooth_tag = "_smooth" if SMOOTHNESS_WEIGHT > 0 else ""
MODEL_OUT = REPO_ROOT / "results" / "models" / f"presence_position_snn_perframe_{_feat_tag}_{_dim_tag}{_smooth_tag}_50ms.pt"

DELTA_THRESHOLD = 0.3
HIDDEN_1 = 128
HIDDEN_2 = 32

LEARNING_RATE = 5e-4
BATCH_SIZE = 32
# 20 epochs wasn't enough -- the 3D run's train loss and val accuracy were
# both still improving at epoch 20, not plateaued. 60 gives real headroom
# to check for convergence instead of guessing.
NUM_EPOCHS = 60
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_dataloaders(X, pos, present, groups, batch_size):
    train_ds, val_ds, test_ds = build_datasets(X, pos, groups, label_dtype=torch.float32, extra=present)
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


def train_one_epoch(model, loader, optimizer, bce_loss, mse_loss, smoothness_weight: float = 0.0) -> float:
    model.train()
    total_loss, total = 0.0, 0
    for windows, pos_seq, pres_seq in loader:
        windows = windows.to(DEVICE)
        pos_seq = pos_seq.to(DEVICE).permute(1, 0, 2)  # (T, batch, out_dim)
        pres_seq = pres_seq.to(DEVICE).permute(1, 0)  # (T, batch)
        optimizer.zero_grad()
        spikes = to_spikes(windows, threshold=DELTA_THRESHOLD)
        pres_pred, pos_pred = model(spikes)
        loss_presence = bce_loss(pres_pred, pres_seq).mean()
        pos_err = ((pos_pred - pos_seq) ** 2).sum(dim=-1)
        loss_position = (pos_err * pres_seq).sum() / (pres_seq.sum() + 1e-8)
        loss = loss_presence + loss_position
        if smoothness_weight > 0:
            both_present = pres_seq[1:] * pres_seq[:-1]  # (T-1, batch) -- only real, in-presence jumps
            jump = ((pos_pred[1:] - pos_pred[:-1]) ** 2).sum(dim=-1)  # (T-1, batch)
            loss_smoothness = (jump * both_present).sum() / (both_present.sum() + 1e-8)
            loss = loss + smoothness_weight * loss_smoothness
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pres_seq.shape[1]
        total += pres_seq.shape[1]
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, bce_loss) -> tuple[float, float, float]:
    """Returns (loss, presence_acc, position_error), all at per-frame granularity."""
    model.eval()
    all_pres_pred, all_pos_pred, all_pres, all_pos = [], [], [], []
    for windows, pos_seq, pres_seq in loader:
        windows = windows.to(DEVICE)
        pos_seq = pos_seq.to(DEVICE).permute(1, 0, 2)
        pres_seq = pres_seq.to(DEVICE).permute(1, 0)
        spikes = to_spikes(windows, threshold=DELTA_THRESHOLD)
        pres_pred, pos_pred = model(spikes)
        all_pres_pred.append(pres_pred)
        all_pos_pred.append(pos_pred)
        all_pres.append(pres_seq)
        all_pos.append(pos_seq)
    pres_pred = torch.cat(all_pres_pred, dim=1)
    pos_pred = torch.cat(all_pos_pred, dim=1)
    pres = torch.cat(all_pres, dim=1)
    pos = torch.cat(all_pos, dim=1)

    presence_acc = ((torch.sigmoid(pres_pred) > 0.5).float() == pres).float().mean().item()
    pos_err = (torch.norm(pos_pred - pos, dim=-1) * pres).sum() / (pres.sum() + 1e-8)
    loss = bce_loss(pres_pred, pres).mean().item() + pos_err.item()
    return loss, presence_acc, pos_err.item()


def main():
    if not RAW_ROOT.exists():
        sys.exit(
            f"No raw captures found at {RAW_ROOT}.\n"
            "See data/raw_capture_loader.py's docstring for the expected layout."
        )
    if USE_3D and not any((RAW_ROOT / d / "videos").exists() for d in [p.name for p in RAW_ROOT.iterdir()]):
        sys.exit(
            "USE_3D=True needs the raw per-trial videos under "
            "data/raw_captures/{condition}_{activity}/videos/ (depth is read off actual "
            "video frames, not just the cached 2D trajectory). None found -- sync them in, "
            "or set USE_3D=False to run the 2D-only per-frame model."
        )

    t_start = time.time()
    print(f"Loading {RATE_MS}ms captures, T_WIN={T_WIN}, STRIDE={STRIDE}, USE_3D={USE_3D}, "
          f"USE_PHASE={USE_PHASE}, SMOOTHNESS_WEIGHT={SMOOTHNESS_WEIGHT}...")
    X, pos, present, groups, activity_codes = load_or_build_perframe_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, DEPTH_CACHE_DIR, CACHE_DIR,
        rate_ms=RATE_MS, t_win=T_WIN, stride=STRIDE, use_3d=USE_3D, use_phase=USE_PHASE,
    )
    print(f"X={X.shape}, pos={pos.shape}, {len(set(groups))} trials ({time.time()-t_start:.1f}s)")

    train_loader, val_loader, test_loader, train_ds = make_dataloaders(X, pos, present, groups, BATCH_SIZE)

    pos_weight = torch.tensor([(1 - train_ds.extra.mean().item()) / train_ds.extra.mean().item()]).to(DEVICE)
    print(f"pos_weight = {pos_weight.item():.3f}")

    model = SNNPresencePositionPerFrame(in_features=X.shape[1] * X.shape[2], h1=HIDDEN_1, h2=HIDDEN_2, out_dim=OUT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    mse_loss = nn.MSELoss(reduction="none")

    best_val_loss, best_state, best_epoch = float("inf"), None, None

    print(f"\nTraining on {DEVICE} for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, bce_loss, mse_loss, SMOOTHNESS_WEIGHT)
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
    print(f"\nFINAL TEST (per-frame, {'3D' if USE_3D else '2D'}): presence_acc={test_presence_acc:.3f}, position_error={test_pos_err:.4f}")
    print(f"total wall time: {time.time()-t_start:.1f}s")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_OUT)
    print(f"saved best checkpoint to {MODEL_OUT}")


if __name__ == "__main__":
    main()
