"""Motion/displacement regression: per-frame conv encoder -> delta-encoded
spikes -> LIF stack -> continuous speed prediction (models.regression_snn).

Only trials with a cached trajectory (data/processed/trajectories/, built by
scripts/build_trajectory_labels.py) are usable -- run that script first (it
may still be running in the background; this script just skips any trial
without a matching .npy yet, so it can be re-run as more trials finish).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from snn_csi_tracking.data import mat_loader
from snn_csi_tracking.data.dataset import build_datasets
from snn_csi_tracking.data.preprocessing import (
    extract_features,
    num_feature_channels,
    sliding_windows,
    speed_labels,
)
from snn_csi_tracking.models.encoding import DeltaEncoder
from snn_csi_tracking.models.regression_snn import CSIConvSpikingRegressor

DATA_ROOT = Path(__file__).parent.parent.parent.parent / "data" / "raw"
CACHE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "processed"
TRAJECTORY_DIR = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "trajectories"
CONDITIONS = ("NLoS", "PLoS")
RATES = (5, 10, 50, 100)  # ground truth now built for all 4
RATE_TO_IDX = {rate_ms: i for i, rate_ms in enumerate(RATES)}

WINDOW_SIZE = 128
STRIDE = 64   
RATE_EMBED_DIM = 8  # the 4 capture rates (5/10/50/100ms) were previously
                     # pooled and thrown away after file discovery, even
                     # though a 128-sample window spans wildly different real
                     # time (0.64s vs 12.8s) depending on rate -- the model
                     # had no way to tell those apart. Embed rate_idx and
                     # concat it onto every timestep's spikes instead.

MAX_NAN_FRAC = 0.3  # exclude non-empty-room trials where MediaPipe mostly
                     # failed to detect anyone (~20 such trials, concentrated
                     # in "L"/"S" -- desk occlusion while seated). Without
                     # this, preprocessing._fill_and_smooth's all-NaN fallback
                     # ("treat as stationary") silently zeroes their speed
                     # labels even though the activity code says someone was
                     # moving -- correct behavior for genuinely-empty ("E")
                     # trials, wrong for these. E trials are exempt from this
                     # filter since 100% NaN is the expected, correct outcome
                     # for them.

# Overridable via env vars for quick sweeps (e.g. SNN_BETA=0.98 .venv/bin/python
# -m snn_csi_tracking.training.train_regression) without hand-editing this file
# per run -- everything defaults to the same values as before if unset.
DELTA_THRESHOLD = float(os.environ.get("SNN_DELTA_THRESHOLD", 0.3))
CONV_CHANNELS = [16, 32]
CONV_KERNEL_SIZE = 9
HIDDEN_SIZES = [128, 32]
BETA = float(os.environ.get("SNN_BETA", 0.9))  # LIF membrane decay
LIF_THRESHOLD = 1.0
DROPOUT = float(os.environ.get("SNN_DROPOUT", 0.0))  

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 32
NUM_EPOCHS = int(os.environ.get("SNN_NUM_EPOCHS", 20))  # best-known config used 20 epochs (see EXPERIMENTS.md)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _discover_trials_with_gt() -> list[tuple[str, int, Path, dict, Path]]:
    """(condition, rate_ms, mat_path, row, trajectory_path) for every trial
    that has BOTH CSI data and a cached ground-truth trajectory with usable
    detection coverage (see MAX_NAN_FRAC)."""
    trials = []
    n_skipped_bad_gt = 0
    for rate_ms in RATES:
        for condition in CONDITIONS:
            path = DATA_ROOT / condition / f"csi_office_{rate_ms}ms_interframe.mat"
            if not path.exists():
                continue
            for row in mat_loader.load_table_metadata(path):
                code = row["activity_code"]
                traj_path = TRAJECTORY_DIR / f"{condition}_{rate_ms}ms_{code}_{row['filename']}.npy"
                if not traj_path.exists():
                    continue
                if code != "E":
                    positions = np.load(traj_path)
                    if np.isnan(positions[:, 0]).mean() > MAX_NAN_FRAC:
                        n_skipped_bad_gt += 1
                        continue
                trials.append((condition, rate_ms, path, row, traj_path))
    if n_skipped_bad_gt:
        print(f"skipped {n_skipped_bad_gt} non-empty trials with >{MAX_NAN_FRAC:.0%} missing position detections")
    return trials


def build_windows_and_labels(dat_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    trials = _discover_trials_with_gt()
    total_trials = len(trials)

    # pass 1: cheap shape peek (CSI metadata only, no CSI/trajectory data
    # read) -> exact total window count and per-trial window count.
    feature_shape = None
    trial_windows = []
    total_windows = 0
    for condition, rate_ms, path, row, traj_path in trials:
        shape = mat_loader.peek_csi_shape(path, row["csi_key"], row["row_index"])
        if shape is None:
            trial_windows.append(0)
            continue
        num_csi, num_subcarriers, num_antennas = shape
        if feature_shape is None:
            feature_shape = (num_feature_channels(num_antennas), num_subcarriers)
        n = max(0, (num_csi - WINDOW_SIZE) // STRIDE + 1)
        trial_windows.append(n)
        total_windows += n

    chan, sub = feature_shape
    gb = total_windows * WINDOW_SIZE * chan * sub * 4 / 1e9
    print(f"total: {total_windows} windows x {WINDOW_SIZE} x {chan} x {sub} "
          f"(~{gb:.1f} GB) -> {dat_path}")

    X = np.memmap(dat_path, dtype="float32", mode="w+", shape=(total_windows, WINDOW_SIZE, chan, sub))
    y = np.empty(total_windows, dtype=np.float32)
    groups = np.empty(total_windows, dtype=np.int64)
    rates = np.empty(total_windows, dtype=np.int64)

    offset = 0
    t_start = time.time()
    n_skipped = 0
    for trial_id, ((condition, rate_ms, path, row, traj_path), n) in enumerate(zip(trials, trial_windows)):
        code = row["activity_code"]
        if n == 0:
            n_skipped += 1
            continue
        csi = mat_loader.load_csi(path, row["csi_key"], row["row_index"])
        if csi is None:
            n_skipped += 1
            continue
        positions = np.load(traj_path)
        if len(positions) != csi.shape[0]:
            print(f"  WARNING: {traj_path.name} has {len(positions)} positions "
                  f"but csi has {csi.shape[0]} frames -- skipping")
            n_skipped += 1
            continue

        feat = extract_features(csi)  # (T, NumSubcarriers, NumChannels)
        feat = feat.transpose(0, 2, 1)  # (T, NumChannels, NumSubcarriers)
        windows = sliding_windows(feat, WINDOW_SIZE, STRIDE)  # (n, W, Chan, Sub)
        labels = speed_labels(positions, WINDOW_SIZE, STRIDE)  # (n,)

        X[offset : offset + n] = windows
        y[offset : offset + n] = labels
        groups[offset : offset + n] = trial_id
        rates[offset : offset + n] = RATE_TO_IDX[rate_ms]
        offset += n

        elapsed = time.time() - t_start
        done = trial_id + 1
        print(
            f"[{done}/{total_trials}] {condition}/{rate_ms}ms/{row['filename']}/{code}: "
            f"{n} windows written (offset={offset}/{total_windows})  "
            f"({elapsed:.1f}s elapsed, ~{elapsed / done * (total_trials - done):.0f}s left)"
        )

    print(f"{total_trials - n_skipped} trials used, {n_skipped} skipped")
    X.flush()
    return X, y, groups, rates


def cache_paths() -> tuple[Path, Path]:
    tag = f"regression_ampphase_rates{'-'.join(str(r) for r in RATES)}_w{WINDOW_SIZE}_s{STRIDE}"
    return CACHE_DIR / f"{tag}.dat", CACHE_DIR / f"{tag}_meta.npz"


def load_or_build_windows() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same caching pattern as train_baseline.py: reuse a disk-backed memmap
    from a previous run if one matching this exact (rates, window, stride)
    config exists, otherwise build it. Delete the two files under
    data/processed/ to force a rebuild (e.g. once more ground truth trials
    finish extracting).
    """
    dat_path, meta_path = cache_paths()
    if dat_path.exists() and meta_path.exists():
        print(f"Loading cached windows from {dat_path}")
        meta = np.load(meta_path)
        X = np.memmap(dat_path, dtype="float32", mode="r", shape=tuple(meta["shape"]))
        return X, meta["y"], meta["groups"], meta["rates"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    X, y, groups, rates = build_windows_and_labels(dat_path)
    np.savez(meta_path, y=y, groups=groups, rates=rates, shape=np.array(X.shape))
    print(f"Cached windows to {dat_path}")
    return X, y, groups, rates


NUM_WORKERS = 8  # overlap disk reads from the memmap with GPU compute --
                 # with num_workers=0 every batch blocks the main process on
                 # a random read from the on-disk array (see dataset.py),
                 # leaving the GPU idle in between


def make_dataloaders(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, rates: np.ndarray, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = build_datasets(X, y, groups, label_dtype=torch.float32, extra=rates)
    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=NUM_WORKERS > 0,
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
    )


def train_one_epoch(model, encoder, loader, optimizer, loss_fn) -> float:
    model.train()
    total_loss, total = 0.0, 0
    for windows, labels, rate_idx in loader:
        windows, labels, rate_idx = windows.to(DEVICE), labels.to(DEVICE), rate_idx.to(DEVICE)
        optimizer.zero_grad()
        pred = model(windows, encoder, rate_idx).squeeze(-1)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, encoder, loader, loss_fn) -> float:
    model.eval()
    total_loss, total = 0.0, 0
    for windows, labels, rate_idx in loader:
        windows, labels, rate_idx = windows.to(DEVICE), labels.to(DEVICE), rate_idx.to(DEVICE)
        pred = model(windows, encoder, rate_idx).squeeze(-1)
        loss = loss_fn(pred, labels)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / total


def main():
    t_start = time.time()
    print(f"Loading {RATES} ms trials with ground truth available...")
    X, y, groups, rates = load_or_build_windows()
    print(f"X={X.shape}, y={y.shape}, {len(set(groups))} trials, "
          f"label range=[{y.min():.2f}, {y.max():.2f}] ({time.time()-t_start:.1f}s)")

    train_loader, val_loader, test_loader = make_dataloaders(X, y, groups, rates, BATCH_SIZE)

    encoder = DeltaEncoder(threshold=DELTA_THRESHOLD).to(DEVICE)
    model = CSIConvSpikingRegressor(
        num_antennas=X.shape[2],  # now amplitude + phase-diff channels, not literal antenna count
        num_subcarriers=X.shape[3],
        conv_channels=CONV_CHANNELS,
        hidden_sizes=HIDDEN_SIZES,
        output_size=1,
        beta=BETA,
        threshold=LIF_THRESHOLD,
        kernel_size=CONV_KERNEL_SIZE,
        num_rates=len(RATES),
        rate_embed_dim=RATE_EMBED_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)
 
    with torch.no_grad():
        sample_windows, _, _ = next(iter(train_loader))
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
        val_loss = evaluate(model, encoder, val_loader, loss_fn)
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            marker = "  <- best so far"
        print(f"epoch {epoch:2d}/{NUM_EPOCHS}  train_mse={train_loss:.4f}  "
              f"val_mse={val_loss:.4f}  ({time.time()-t0:.1f}s){marker}")

    # early stopping: evaluate on the checkpoint with the best val loss 
    print(f"\nrestoring best checkpoint from epoch {best_epoch} (val_mse={best_val_loss:.4f})")
    model.load_state_dict(best_state)

    test_loss = evaluate(model, encoder, test_loader, loss_fn)
    test_label_var = test_loader.dataset.labels.numpy().var()
    print(f"test_mse={test_loss:.4f}  (test-set label variance={test_label_var:.4f}, "
          f"so R^2 = {1 - test_loss / test_label_var:.3f})")
    print(f"total wall time: {time.time()-t_start:.1f}s")

    model_out = CACHE_DIR.parent / "results" / "models" / "regression_snn.pt"
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_out)
    print(f"saved best checkpoint to {model_out}")


if __name__ == "__main__":
    main()
