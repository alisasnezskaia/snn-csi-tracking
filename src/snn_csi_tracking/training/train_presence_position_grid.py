"""Coarse grid-cell classification for the presence_position pipeline: same
per-frame LIF trunk as train_presence_position_perframe.py (SNNPresence
PositionPerFrame), but the position head predicts which of GRID_DIM**2
discretized floor cells the person is in, instead of continuous (x, y).

Why: plain continuous (x, y) regression on this pipeline's own data
collapses to near-constant predictions (see results/figures/
14_trajectory_prediction.png) and scored weakly (R^2=0.158) on a separate
diagnostic (see train_grid_motion.py's docstring) -- the fix that actually
worked there wasn't a better model, it was reformulating the target from
exact coordinates to a coarser, more forgiving one. This script runs the
same experiment for this pipeline: is coarse, per-timestep localization
achievable where exact position isn't?

Also z-scores amplitude against the empty-room baseline (raw_capture_loader.
load_empty_room_baseline via presence_position_dataset.compute_features's
`baseline` arg) instead of each trial's own mean/std -- a genuinely
person-free static-clutter reference should raise SNR for this task too,
not just presence detection (see conversation).

Two more features were tried on top of the first grid-cell run (test
grid_acc=0.394 vs majority-baseline=0.316), both meant to recover
information the frequency-domain amplitude/phase features structurally
discard (see conversation):
  - USE_CIR: channel impulse response (IFFT across subcarriers) alongside
    the frequency-domain amplitude/phase, so multipath reflections are
    separated by delay/path-length instead of only seen as their tangled
    per-subcarrier sum (presence_position_dataset.compute_features's
    `use_cir` arg).
  - USE_DELTA_RATE: the binary delta/spike encoding (to_spikes) only
    encodes *whether* a subcarrier's amplitude changed past threshold each
    frame, discarding *how much* and in *which direction* -- exactly where
    motion-direction/speed (Doppler-like) information would live.
    to_spikes_and_delta additionally returns that signed, clipped raw delta
    as a continuous companion channel, concatenated alongside the spikes
    (doubling in_features).

BOTH DEFAULT TO FALSE: tried together (test grid_acc=0.013), CIR alone
(0.000), and delta-rate alone (0.021) -- all *worse* than chance, let alone
the 0.394 ampphase-only baseline, while presence_acc stayed fine (~0.83-
0.87) in every case. Training loss kept dropping well below the working
run's (~0.11 vs ~0.19 at a comparable epoch) while grid accuracy collapsed
-- a plain overfitting signature, not evidence the underlying ideas are
wrong: CIR adds 3 full 1024-wide channels and delta-rate doubles the whole
feature width, both directly enlarging fc1's weight count (a plain
Linear, no dropout) against only ~30 training trials. Before retrying
either, they likely need either much lower dimensionality (e.g. truncate
CIR to a handful of leading delay taps instead of the full 1024, summarize
delta-rate as a few bands/statistics instead of per-subcarrier) or added
regularization (dropout, weight decay) -- not simply re-enabling as-is.

AoA (MUSIC angle-of-arrival) was tried and dropped entirely (not just
defaulted off) -- checked empirically (see conversation): correlation
between the estimated angle and true x-position flips sign across trials
(-0.12 to +0.29), and the eigenvalue ratio (2nd/1st, near-0 if single-
source MUSIC's core assumption held) averages 0.26-0.52 across trials,
i.e. a second comparably-strong path is present almost everywhere. Not a
calibration bug -- single-source MUSIC's assumption is structurally
violated by real multipath with only 3 antennas.

Uses the held-out-trial split (dataset.build_datasets_grouped), not the
chronological within-trial split train_presence_position_perframe.py still
uses -- this is exactly the kind of result that split matters for (a model
could otherwise key off one trial's session-specific fingerprint instead of
learning the general cell mapping), same reasoning as train_grid_motion.py.
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

from snn_csi_tracking.data.dataset import build_datasets_grouped
from snn_csi_tracking.data.presence_position_dataset import load_or_build_perframe_grid_dataset
from snn_csi_tracking.models.presence_position_snn import SNNPresencePositionPerFrame, to_spikes_and_delta

REPO_ROOT = Path(__file__).parent.parent.parent.parent
RAW_ROOT = REPO_ROOT / "data" / "raw_captures"
TRAJECTORY_CACHE_DIR = REPO_ROOT / "data" / "processed" / "presence_position_trajectories"
CACHE_DIR = REPO_ROOT / "data" / "processed"

RATE_MS = 50
T_WIN = 64
STRIDE = 32
GRID_DIM = 3  # 3x3 = 9 cells -- coarse first pass (train_grid_motion.py's
              # precedent: "start coarse, only go finer once this holds up")
USE_PHASE = True
USE_EMPTY_BASELINE = True
USE_CIR = False
USE_DELTA_RATE = False
DENOISE = "wavelet"  # None, "wavelet", or "pca" -- see preprocessing.wavelet_denoise/
                # pca_denoise. Applied to amplitude before the empty-room
                # baseline z-scoring (both the live signal and the baseline
                # reference itself, consistently -- see conversation on why
                # most CSI position/localization work in the literature
                # denoises first).

_tag = "_ebase" if USE_EMPTY_BASELINE else ""
_tag += "_cir" if USE_CIR else ""
_tag += "_delta" if USE_DELTA_RATE else ""
_tag += f"_{DENOISE}denoise" if DENOISE else ""
MODEL_OUT = REPO_ROOT / "results" / "models" / f"presence_position_grid{GRID_DIM}{_tag}_50ms.pt"

DELTA_THRESHOLD = 0.3
HIDDEN_1 = 128
HIDDEN_2 = 32

LEARNING_RATE = 5e-4
BATCH_SIZE = 32
NUM_EPOCHS = 60
NUM_WORKERS = 4
SEED = 0  # only ~30/10/10 train/val/test trials -- results vary a lot run
          # to run from weight init / loader shuffling alone (see
          # conversation), so a fixed seed is required for any config
          # comparison to mean anything; sweep --seed for a real estimate.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_dataloaders(X, cells, present, groups, activity_codes, batch_size):
    train_ds, val_ds, test_ds = build_datasets_grouped(
        X, cells, groups, label_dtype=torch.long, extra=present, strata=activity_codes,
    )
    print(f"split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} windows, "
          f"train_trials={len(set(groups[train_ds.indices]))} "
          f"val_trials={len(set(groups[val_ds.indices]))} "
          f"test_trials={len(set(groups[test_ds.indices]))}")
    loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), persistent_workers=NUM_WORKERS > 0)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        train_ds, test_ds,
    )


def encode(windows: torch.Tensor) -> torch.Tensor:
    """Spike encoding fed to the model: binary delta/spikes, optionally
    concatenated with the continuous signed delta itself (see
    USE_DELTA_RATE / to_spikes_and_delta's docstring)."""
    spikes, delta = to_spikes_and_delta(windows, threshold=DELTA_THRESHOLD)
    if USE_DELTA_RATE:
        return torch.cat([spikes, delta], dim=-1)
    return spikes


def train_one_epoch(model, loader, optimizer, bce_loss, ce_loss) -> float:
    model.train()
    total_loss, total = 0.0, 0
    for windows, cell_seq, pres_seq in loader:
        windows = windows.to(DEVICE)
        cell_seq = cell_seq.to(DEVICE).permute(1, 0)  # (T, batch)
        pres_seq = pres_seq.to(DEVICE).permute(1, 0)  # (T, batch)
        optimizer.zero_grad()
        x_seq = encode(windows)
        pres_pred, cell_logits = model(x_seq)  # cell_logits: (T, batch, K)
        loss_presence = bce_loss(pres_pred, pres_seq).mean()
        k = cell_logits.shape[-1]
        ce_per_frame = ce_loss(cell_logits.reshape(-1, k), cell_seq.reshape(-1)).reshape(cell_seq.shape)
        loss_grid = (ce_per_frame * pres_seq).sum() / (pres_seq.sum() + 1e-8)
        loss = loss_presence + loss_grid
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pres_seq.shape[1]
        total += pres_seq.shape[1]
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, bce_loss, ce_loss) -> tuple[float, float, float]:
    """Returns (loss, presence_acc, grid_acc) -- grid_acc is top-1 cell
    accuracy, computed only over frames marked present in the ground truth."""
    model.eval()
    all_pres_pred, all_cell_logits, all_pres, all_cells = [], [], [], []
    for windows, cell_seq, pres_seq in loader:
        windows = windows.to(DEVICE)
        cell_seq = cell_seq.to(DEVICE).permute(1, 0)
        pres_seq = pres_seq.to(DEVICE).permute(1, 0)
        x_seq = encode(windows)
        pres_pred, cell_logits = model(x_seq)
        all_pres_pred.append(pres_pred)
        all_cell_logits.append(cell_logits)
        all_pres.append(pres_seq)
        all_cells.append(cell_seq)
    pres_pred = torch.cat(all_pres_pred, dim=1)
    cell_logits = torch.cat(all_cell_logits, dim=1)
    pres = torch.cat(all_pres, dim=1)
    cells = torch.cat(all_cells, dim=1)

    presence_acc = ((torch.sigmoid(pres_pred) > 0.5).float() == pres).float().mean().item()
    pred_cell = cell_logits.argmax(dim=-1)
    grid_acc = ((pred_cell == cells).float() * pres).sum().item() / (pres.sum().item() + 1e-8)

    k = cell_logits.shape[-1]
    ce_per_frame = ce_loss(cell_logits.reshape(-1, k), cells.reshape(-1)).reshape(cells.shape)
    loss_grid = (ce_per_frame * pres).sum() / (pres.sum() + 1e-8)
    loss = bce_loss(pres_pred, pres).mean().item() + loss_grid.item()
    return loss, presence_acc, grid_acc


def majority_baseline_acc(train_ds, test_ds, k: int) -> tuple[float, int]:
    """Accuracy of always predicting train's single most common present
    cell, evaluated on test -- the honest floor any real model must clear."""
    train_present = train_ds.extra.bool()
    majority_cell = torch.bincount(train_ds.labels[train_present], minlength=k).argmax().item()
    test_present = test_ds.extra.bool()
    test_cells = test_ds.labels[test_present]
    acc = (test_cells == majority_cell).float().mean().item()
    return acc, majority_cell


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--denoise", choices=["none", "wavelet", "pca"], default=(DENOISE or "none"))
    parser.add_argument("--seed", type=int, default=SEED,
                         help="torch/numpy seed for weight init + loader shuffling (data split stays fixed at seed=0)")
    args = parser.parse_args()
    denoise = None if args.denoise == "none" else args.denoise

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not RAW_ROOT.exists():
        sys.exit(
            f"No raw captures found at {RAW_ROOT}.\n"
            "See data/raw_capture_loader.py's docstring for the expected layout."
        )

    t_start = time.time()
    k = GRID_DIM * GRID_DIM
    print(f"Loading {RATE_MS}ms captures, T_WIN={T_WIN}, STRIDE={STRIDE}, GRID_DIM={GRID_DIM} ({k} cells), "
          f"USE_EMPTY_BASELINE={USE_EMPTY_BASELINE}, USE_CIR={USE_CIR}, USE_DELTA_RATE={USE_DELTA_RATE}, "
          f"denoise={denoise}, seed={args.seed}...")
    X, cells, present, groups, activity_codes = load_or_build_perframe_grid_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, CACHE_DIR, GRID_DIM,
        rate_ms=RATE_MS, t_win=T_WIN, stride=STRIDE, use_phase=USE_PHASE,
        use_empty_baseline=USE_EMPTY_BASELINE, use_cir=USE_CIR, denoise=denoise,
    )
    print(f"X={X.shape}, cells={cells.shape}, {len(set(groups))} trials ({time.time()-t_start:.1f}s)")

    train_loader, val_loader, test_loader, train_ds, test_ds = make_dataloaders(
        X, cells, present, groups, activity_codes, BATCH_SIZE
    )

    chance_acc = 1.0 / k
    test_majority_acc, majority_cell = majority_baseline_acc(train_ds, test_ds, k)
    print(f"chance accuracy (1/{k}): {chance_acc:.3f}")
    print(f"majority-cell baseline (always predict train's cell {majority_cell}): test acc={test_majority_acc:.3f}")

    present_mask = train_ds.extra.bool()
    present_cells = train_ds.labels[present_mask]
    class_counts = torch.bincount(present_cells, minlength=k).float()
    print(f"train cell distribution (present frames only): {class_counts.tolist()}")
    class_weights = (class_counts.sum() / (class_counts + 1e-6))
    class_weights = (class_weights / class_weights.mean()).to(DEVICE)

    train_presence_rate = train_ds.extra.float().mean().item()
    pos_weight = torch.tensor([(1 - train_presence_rate) / train_presence_rate]).to(DEVICE)
    print(f"pos_weight (presence) = {pos_weight.item():.3f}")

    in_features = X.shape[1] * X.shape[2] * (2 if USE_DELTA_RATE else 1)
    model = SNNPresencePositionPerFrame(in_features=in_features, h1=HIDDEN_1, h2=HIDDEN_2, out_dim=k).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    # Selected by val_grid_acc directly, not the combined BCE+CE val_loss --
    # that loss can keep dropping (presence loss dominating, or CE improving
    # in a way loss-magnitude doesn't reflect) while grid accuracy is still
    # rising, picking an undertrained early checkpoint instead of the model
    # that actually classifies cells best (same failure mode documented in
    # train_grid_motion.py's evaluate() docstring, there for an analogous
    # log-space-vs-raw-space mismatch).
    best_val_grid_acc, best_state, best_epoch = -1.0, None, None

    print(f"\nTraining on {DEVICE} for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, bce_loss, ce_loss)
        val_loss, val_presence_acc, val_grid_acc = evaluate(model, val_loader, bce_loss, ce_loss)
        marker = ""
        if val_grid_acc > best_val_grid_acc:
            best_val_grid_acc = val_grid_acc
            best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
            best_epoch = epoch
            marker = "  <- best so far"
        print(
            f"epoch {epoch:2d}/{NUM_EPOCHS}  train_loss={train_loss:.5f}  "
            f"val_presence_acc={val_presence_acc:.3f}  val_grid_acc={val_grid_acc:.3f} (chance={chance_acc:.3f})  "
            f"({time.time()-t0:.1f}s){marker}"
        )

    print(f"\nrestoring best checkpoint from epoch {best_epoch} (val_grid_acc={best_val_grid_acc:.3f})")
    model.load_state_dict(best_state)

    _test_loss, test_presence_acc, test_grid_acc = evaluate(model, test_loader, bce_loss, ce_loss)
    print(f"\nFINAL TEST: presence_acc={test_presence_acc:.3f}, grid_acc={test_grid_acc:.3f}  "
          f"(chance={chance_acc:.3f}, majority-baseline={test_majority_acc:.3f})")
    print(f"total wall time: {time.time()-t_start:.1f}s")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_OUT)
    print(f"saved best checkpoint to {MODEL_OUT}")


if __name__ == "__main__":
    main()
