"""Coarse spatial motion regression: same amplitude+phase -> per-frame conv
-> delta-encoded spikes -> LIF stack pipeline as train_regression.py, but the
readout predicts a K-dim per-grid-cell motion vector instead of one scalar --
"how much motion, and roughly where" (see conversation notes: plain
continuous (x,y) position regression scored weakly elsewhere, R^2=0.158;
this tests a coarser, more forgiving spatial target instead, after a
follow-up sanity check showed the diagnostic tool used to rule out grid
localization -- PCA+KNN on hand-summarized features -- was itself too weak
to even detect presence, an easy, well-established effect; this script
instead trains the real architecture end to end).

Single rate (50ms) only, non-overlapping 5-second windows: keeps this first
attempt simple (fixed window length in frames, no rate-conditioning needed)
-- mixing capture rates isn't straightforward once windows are defined by
real duration rather than a fixed frame count, since PerFrameConvEncoder's
downstream LIF loop needs a fixed number of timesteps per batch.

Uses the held-out-trial split (dataset.build_datasets_grouped), not the
chronological within-trial split train_regression.py still uses -- this
script is exactly the case that split matters for (see conversation notes
on why a within-trial split lets a model key off session-specific
fingerprints instead of learning the general mapping).

Only trials with a cached trajectory (data/processed/trajectories/, built by
scripts/build_trajectory_labels.py) are usable.

Two more fixes on top of the first (undertrained) run's diagnosis: the
model is trained on log1p(label) rather than the raw pixel-displacement
label (MSE on the raw, heavy-tailed scale let a few large-displacement
cells dominate the loss), and training windows are oversampled by how much
motion they contain (most windows are near-all-zero -- empty room, or
someone sitting still -- which would otherwise let the model get away with
just predicting near-zero everywhere). See conversation notes.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from snn_csi_tracking.data import mat_loader
from snn_csi_tracking.data.dataset import build_datasets_grouped
from snn_csi_tracking.data.preprocessing import extract_features, grid_motion_labels, sliding_windows
from snn_csi_tracking.models.encoding import DeltaEncoder
from snn_csi_tracking.models.regression_snn import CSIConvSpikingRegressor

DATA_ROOT = Path(__file__).parent.parent.parent.parent / "data" / "raw"
CACHE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "processed"
TRAJECTORY_DIR = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "trajectories"
CONDITIONS = ("NLoS", "PLoS")
RATE_MS = 50

WINDOW_SECONDS = 5.0
WINDOW_SIZE = round(WINDOW_SECONDS * 1000 / RATE_MS)  # 100 frames at 50ms
STRIDE = WINDOW_SIZE  # non-overlapping -- avoids near-duplicate windows
                       # biasing the held-out-trial split's test set (see
                       # EXPERIMENTS.md run #1 / diagnose_position_umap.py)
GRID_DIM = 2  # 2x2 = 4 cells -- start coarse; only go finer once this holds
              # up (the earlier PCA+KNN diagnostic's negative result at this
              # same resolution wasn't trustworthy, see conversation notes)

MAX_NAN_FRAC = 0.3  # see train_regression.py -- excludes non-empty trials
                     # where pose detection mostly failed

DELTA_THRESHOLD = 0.3
CONV_CHANNELS = [16, 32]
CONV_KERNEL_SIZE = 9
HIDDEN_SIZES = [128, 32]
BETA = 0.9
LIF_THRESHOLD = 1.0
DROPOUT = 0.0

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 16  # small dataset (a few hundred windows) -- keep batches modest
NUM_EPOCHS = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _discover_trials_with_gt() -> list[tuple[str, Path, dict, Path]]:
    """(condition, mat_path, row, trajectory_path) for every RATE_MS trial
    with both CSI data and a usable cached ground-truth trajectory. Same
    filtering convention as train_regression.py's version of this function."""
    trials = []
    n_skipped_bad_gt = 0
    for condition in CONDITIONS:
        path = DATA_ROOT / condition / f"csi_office_{RATE_MS}ms_interframe.mat"
        if not path.exists():
            continue
        for row in mat_loader.load_table_metadata(path):
            code = row["activity_code"]
            traj_path = TRAJECTORY_DIR / f"{condition}_{RATE_MS}ms_{code}_{row['filename']}.npy"
            if not traj_path.exists():
                continue
            if code != "E":
                positions = np.load(traj_path)
                if np.isnan(positions[:, 0]).mean() > MAX_NAN_FRAC:
                    n_skipped_bad_gt += 1
                    continue
            trials.append((condition, path, row, traj_path))
    if n_skipped_bad_gt:
        print(f"skipped {n_skipped_bad_gt} non-empty trials with >{MAX_NAN_FRAC:.0%} missing position detections")
    return trials


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, groups, activity_codes):
    X: (NumWindows, WINDOW_SIZE, NumChannels, NumSubcarriers) float32
    y: (NumWindows, GRID_DIM**2) float32 -- per-cell motion magnitude
    groups: (NumWindows,) int64 trial id, for the held-out-trial split
    activity_codes: (NumWindows,) str, used to stratify that split
    """
    trials = _discover_trials_with_gt()
    print(f"{len(trials)} usable {RATE_MS}ms trials")

    # bounds pass: trajectories only (cheap), before touching any CSI --
    # grid cells must mean the same physical region for every trial.
    all_x, all_y = [], []
    for condition, path, row, traj_path in trials:
        if row["activity_code"] == "E":
            continue
        positions = np.load(traj_path)
        valid = ~np.isnan(positions[:, 0])
        if valid.any():
            all_x.append(positions[valid, 0])
            all_y.append(positions[valid, 1])
    bounds = (
        np.concatenate(all_x).min(), np.concatenate(all_x).max(),
        np.concatenate(all_y).min(), np.concatenate(all_y).max(),
    )
    print(f"floor bounds (pixel space, uncalibrated): x=[{bounds[0]:.0f},{bounds[1]:.0f}] "
          f"y=[{bounds[2]:.0f},{bounds[3]:.0f}]")

    X_list, y_list, groups, activity_codes = [], [], [], []
    t_start = time.time()
    for trial_id, (condition, path, row, traj_path) in enumerate(trials):
        csi = mat_loader.load_csi(path, row["csi_key"], row["row_index"])
        if csi is None:
            continue
        positions = np.load(traj_path)
        if len(positions) != csi.shape[0]:
            print(f"  WARNING: {traj_path.name} frame-count mismatch -- skipping")
            continue

        feat = extract_features(csi)  # (T, Sub, Chan)
        feat = feat.transpose(0, 2, 1)  # (T, Chan, Sub)
        windows = sliding_windows(feat, WINDOW_SIZE, STRIDE)  # (n, W, Chan, Sub)
        labels = grid_motion_labels(positions, bounds, GRID_DIM, WINDOW_SIZE, STRIDE)  # (n, K)

        X_list.append(windows)
        y_list.append(labels)
        groups.extend([trial_id] * len(windows))
        activity_codes.extend([row["activity_code"]] * len(windows))

        elapsed = time.time() - t_start
        print(f"[{trial_id+1}/{len(trials)}] {condition}/{row['filename']}/{row['activity_code']}: "
              f"{len(windows)} windows  ({elapsed:.1f}s elapsed)")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.array(groups, dtype=np.int64)
    activity_codes = np.array(activity_codes)
    return X, y, groups, activity_codes


def cache_paths() -> tuple[Path, Path]:
    tag = f"gridmotion_{RATE_MS}ms_grid{GRID_DIM}_w{WINDOW_SIZE}_s{STRIDE}"
    return CACHE_DIR / f"{tag}.npz", CACHE_DIR / f"{tag}_X.npy"


def load_or_build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    meta_path, x_path = cache_paths()
    if meta_path.exists() and x_path.exists():
        print(f"Loading cached dataset from {x_path}")
        meta = np.load(meta_path, allow_pickle=True)
        X = np.load(x_path, mmap_mode="r")
        return X, meta["y"], meta["groups"], meta["activity_codes"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    X, y, groups, activity_codes = build_dataset()
    meta_path, x_path = cache_paths()
    np.save(x_path, X)
    np.savez(meta_path, y=y, groups=groups, activity_codes=activity_codes)
    print(f"Cached dataset to {x_path}")
    return X, y, groups, activity_codes


NUM_WORKERS = 4


def make_dataloaders(X, y, groups, activity_codes, batch_size):
    train_ds, val_ds, test_ds = build_datasets_grouped(
        X, y, groups, label_dtype=torch.float32, strata=activity_codes
    )
    print(f"split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} windows, "
          f"train_trials={len(set(groups[train_ds.indices]))} "
          f"val_trials={len(set(groups[val_ds.indices]))} "
          f"test_trials={len(set(groups[test_ds.indices]))}")
    loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
                          persistent_workers=NUM_WORKERS > 0)

    # Oversample windows with real spatial motion during training -- most
    # windows are near-all-zero (empty room, person sitting still), which
    # would otherwise dominate the gradient signal and let the model get
    # away with just predicting near-zero everywhere (see conversation
    # notes). val/test stay on the natural, unweighted distribution so the
    # reported metric still reflects real-world class balance, not a
    # rebalanced one.
    train_weights = train_ds.labels.sum(dim=1).sqrt() + 0.1
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

    return (
        DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **loader_kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
    )


def train_one_epoch(model, encoder, loader, optimizer, loss_fn) -> float:
    """The model is trained to predict log1p(label), not the raw pixel-
    displacement label -- MSE on the raw scale lets a handful of large-
    displacement cells dominate the loss and drown out the gradient signal
    from the many near-zero cells (see conversation notes)."""
    model.train()
    total_loss, total = 0.0, 0
    for windows, labels in loader:
        windows, labels = windows.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        pred = model(windows, encoder)
        loss = loss_fn(pred, torch.log1p(labels))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, encoder, loader) -> tuple[float, float]:
    """Returns (raw-space mse, raw-space r2): predictions are inverted back
    to raw pixel-displacement units (expm1) before comparing against the
    untransformed labels. Deliberately NOT the log-space loss the model is
    trained on -- checkpoint selection needs to compare against the metric
    we actually report and care about. Using the training-space (log) loss
    for selection let a checkpoint from epoch 6/200 -- essentially
    undertrained -- get picked as "best", because log-space and raw-space
    validation error diverged after the first few epochs (see conversation
    notes): as training (with oversampled motion-heavy batches) improved
    raw-space accuracy on the rare true-motion cells, it likely got very
    slightly worse at predicting exact zero for the many empty cells in the
    natural, unweighted validation set, which the log-space loss --
    evaluated on that same natural, zero-heavy set -- penalizes more than
    raw MSE does."""
    model.eval()
    all_pred, all_true = [], []
    for windows, labels in loader:
        windows, labels = windows.to(DEVICE), labels.to(DEVICE)
        pred = model(windows, encoder)
        all_pred.append(torch.expm1(pred).cpu().numpy())
        all_true.append(labels.cpu().numpy())
    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    mse = ((all_true - all_pred) ** 2).mean()
    ss_res = ((all_true - all_pred) ** 2).sum()
    ss_tot = ((all_true - all_true.mean(axis=0, keepdims=True)) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return mse, r2


def main():
    t_start = time.time()
    print(f"Loading {RATE_MS}ms trials, {GRID_DIM}x{GRID_DIM} grid, {WINDOW_SECONDS:.0f}s windows...")
    X, y, groups, activity_codes = load_or_build_dataset()
    print(f"X={X.shape}, y={y.shape}, {len(set(groups))} trials ({time.time()-t_start:.1f}s)")

    train_loader, val_loader, test_loader = make_dataloaders(X, y, groups, activity_codes, BATCH_SIZE)

    k = GRID_DIM * GRID_DIM
    encoder = DeltaEncoder(threshold=DELTA_THRESHOLD).to(DEVICE)
    model = CSIConvSpikingRegressor(
        num_antennas=X.shape[2],  # amplitude+phase channels, not literal antennas -- see extract_features
        num_subcarriers=X.shape[3],
        conv_channels=CONV_CHANNELS,
        hidden_sizes=HIDDEN_SIZES,
        output_size=k,
        beta=BETA,
        threshold=LIF_THRESHOLD,
        kernel_size=CONV_KERNEL_SIZE,
        num_rates=None,
        dropout=DROPOUT,
    ).to(DEVICE)

    with torch.no_grad():
        sample_windows, _ = next(iter(train_loader))
        sample_features = model.conv_encoder(sample_windows.to(DEVICE))
        sample_spikes = encoder(sample_features)
        print(f"spike sparsity at threshold={DELTA_THRESHOLD}: "
              f"{sample_spikes.mean().item():.4f} fraction of entries fire")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = None

    print(f"\nTraining on {DEVICE} for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, encoder, train_loader, optimizer, loss_fn)
        val_loss, val_r2 = evaluate(model, encoder, val_loader)
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
            best_epoch = epoch
            marker = "  <- best so far"
        print(f"epoch {epoch:2d}/{NUM_EPOCHS}  train_mse={train_loss:.4f}  "
              f"val_mse={val_loss:.4f}  val_r2={val_r2:.3f}  ({time.time()-t0:.1f}s){marker}")

    print(f"\nrestoring best checkpoint from epoch {best_epoch} (val_mse={best_val_loss:.4f})")
    model.load_state_dict(best_state)

    test_mse, test_r2 = evaluate(model, encoder, test_loader)
    print(f"test_mse={test_mse:.4f}  test_r2={test_r2:.3f}")
    print(f"total wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
