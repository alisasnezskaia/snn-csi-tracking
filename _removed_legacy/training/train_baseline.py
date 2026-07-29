from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from snn_csi_tracking.data import mat_loader
from snn_csi_tracking.data.dataset import build_datasets
from snn_csi_tracking.data.preprocessing import normalize, sliding_windows
from snn_csi_tracking.models.encoding import DeltaEncoder
from snn_csi_tracking.models.snn import CSISpikingNet

DATA_ROOT = Path(__file__).parent.parent.parent.parent / "data" / "raw"
CACHE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "processed"
CONDITIONS = ("NLoS", "PLoS")
 
RATES = (5, 10, 50, 100)

# How many consecutive CSI frames make one training sample, and how far the
# window slides between samples (preprocessing.sliding_windows). Bigger
# window = more temporal context per sample but fewer samples total, and a
# longer spike train for the SNN to process per forward pass.
WINDOW_SIZE = 128
STRIDE = 128
 
DELTA_THRESHOLD = 0.3

HIDDEN_SIZES = [256, 64]
BETA = 0.9  # LIF membrane decay
LIF_THRESHOLD = 1.0  # LIF firing threshold, snntorch default

# Standard training knobs.
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 10 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def _discover_trials(rates: tuple[int, ...]) -> list[tuple[int, str, Path, dict]]:
    """Metadata-only pass: which (rate, condition, path, row) trials exist."""
    label_to_idx = {c: i for i, c in enumerate(mat_loader.ACTIVITY_CODES)}
    trials = []
    for rate_ms in rates:
        for condition in CONDITIONS:
            path = DATA_ROOT / condition / f"csi_office_{rate_ms}ms_interframe.mat"
            if not path.exists():
                continue
            for row in mat_loader.load_table_metadata(path):
                if row["activity_code"] in label_to_idx:
                    trials.append((rate_ms, condition, path, row))
    return trials


def build_windows_and_labels(
    rates: tuple[int, ...], window_size: int, stride: int, dat_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Write every trial's windows directly into a np.memmap on disk at
    `dat_path`, instead of collecting per-trial arrays in a Python list and
    np.concatenate-ing at the end -- that pattern needs roughly 2x the final
    size in RAM at once (the list of pieces + the fresh concatenated copy),
    which doesn't fit in memory once all 4 interframe rates are pooled.

    Two passes: first, cheap metadata-only peeks (mat_loader.peek_csi_shape,
    no CSI data read) to compute how many windows *every* trial will produce
    and thus the exact total size to preallocate; second, the real per-trial
    load + window + write-into-memmap loop.

    Returns (X, y, groups) with X a disk-backed memmap of shape
    (NumWindows, window_size, NumSubcarriers*NumAntennas).
    """
    label_to_idx = {c: i for i, c in enumerate(mat_loader.ACTIVITY_CODES)}
    trials = _discover_trials(rates)
    total_trials = len(trials)

    # pass 1: cheap shape peek per trial -> exact total window count, and how
    # many windows each trial contributes (so pass 2 knows where to write).
    feature_dim = None
    trial_windows = []
    total_windows = 0
    for rate_ms, condition, path, row in trials:
        shape = mat_loader.peek_csi_shape(path, row["csi_key"], row["row_index"])
        if shape is None:
            trial_windows.append(0)
            continue
        num_csi, num_subcarriers, num_antennas = shape
        if feature_dim is None:
            feature_dim = num_subcarriers * num_antennas
        n = max(0, (num_csi - window_size) // stride + 1)
        trial_windows.append(n)
        total_windows += n

    gb = total_windows * window_size * feature_dim * 4 / 1e9
    print(f"total: {total_windows} windows x {window_size} x {feature_dim} "
          f"(~{gb:.1f} GB) -> {dat_path}")

    X = np.memmap(dat_path, dtype="float32", mode="w+", shape=(total_windows, window_size, feature_dim))
    y = np.empty(total_windows, dtype=np.int64)
    groups = np.empty(total_windows, dtype=np.int64)

    # pass 2: load one trial at a time, window it, write straight into X's
    # on-disk buffer -- never holds more than one trial's raw CSI (a few
    # hundred MB at most) plus the fixed-size X buffer, no list of copies.
    offset = 0
    t_start = time.time()
    for trial_id, ((rate_ms, condition, path, row), n) in enumerate(zip(trials, trial_windows)):
        code = row["activity_code"]
        if n == 0:
            print(f"[{trial_id + 1}/{total_trials}] {condition}/{rate_ms}ms/{row['filename']}/{code}: "
                  f"SKIPPED (placeholder row)")
            continue
        csi = mat_loader.load_csi(path, row["csi_key"], row["row_index"])
        amp = normalize(csi)  # (T, NumSubcarriers, NumAntennas)
        amp = amp.reshape(amp.shape[0], -1)  # (T, feature_dim)
        windows = sliding_windows(amp, window_size, stride)  # (n, window_size, feature_dim)

        X[offset : offset + n] = windows
        y[offset : offset + n] = label_to_idx[code]
        groups[offset : offset + n] = trial_id
        offset += n

        elapsed = time.time() - t_start
        done = trial_id + 1
        print(
            f"[{done}/{total_trials}] {condition}/{rate_ms}ms/{row['filename']}/{code}: "
            f"{n} windows written (offset={offset}/{total_windows})  "
            f"({elapsed:.1f}s elapsed, ~{elapsed / done * (total_trials - done):.0f}s left)"
        )

    X.flush()
    return X, y, groups


def cache_paths(rates: tuple[int, ...], window_size: int, stride: int) -> tuple[Path, Path]:
    tag = f"rates{'-'.join(str(r) for r in rates)}_w{window_size}_s{stride}"
    return CACHE_DIR / f"windows_{tag}.dat", CACHE_DIR / f"windows_{tag}_meta.npz"


def load_or_build_windows(
    rates: tuple[int, ...], window_size: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reuse a cached (X, y, groups) from disk if one matching this exact
    (rates, window_size, stride) config exists; otherwise build it (writing
    the big array straight to disk via memmap, see build_windows_and_labels)
    and cache it for next time.

    X stays a disk-backed memmap (never fully loaded into RAM at once); y and
    groups are tiny and kept as plain in-memory arrays, stored in a small
    companion _meta.npz alongside the raw .dat file.

    Delete both files under data/processed/ to force a rebuild (e.g. after
    the underlying .mat files themselves change).
    """
    dat_path, meta_path = cache_paths(rates, window_size, stride)
    if dat_path.exists() and meta_path.exists():
        print(f"Loading cached windows from {dat_path}")
        meta = np.load(meta_path)
        X = np.memmap(dat_path, dtype="float32", mode="r", shape=tuple(meta["shape"]))
        return X, meta["y"], meta["groups"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    X, y, groups = build_windows_and_labels(rates, window_size, stride, dat_path)
    np.savez(meta_path, y=y, groups=groups, shape=np.array(X.shape))
    print(f"Cached windows to {dat_path}")
    return X, y, groups


def make_dataloaders(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Wraps build_datasets() (chronological per-trial train/val/test split) in DataLoaders."""
    train_ds, val_ds, test_ds = build_datasets(X, y, groups)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def train_one_epoch(
    model: nn.Module,
    encoder: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for windows, labels in loader:
        windows, labels = windows.to(DEVICE), labels.to(DEVICE)
        spikes = encoder(windows)

        optimizer.zero_grad()
        logits = model(spikes)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module, encoder: nn.Module, loader: DataLoader, loss_fn: nn.Module
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for windows, labels in loader:
        windows, labels = windows.to(DEVICE), labels.to(DEVICE)
        spikes = encoder(windows)
        logits = model(spikes)
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main():
    t_start = time.time()
    print(f"Loading {RATES} ms trials...")
    X, y, groups = load_or_build_windows(RATES, WINDOW_SIZE, STRIDE)
    print(f"X={X.shape}, y={y.shape}, {len(set(groups))} trials "
          f"({time.time()-t_start:.1f}s)")

    train_loader, val_loader, test_loader = make_dataloaders(X, y, groups, BATCH_SIZE)

    encoder = DeltaEncoder(threshold=DELTA_THRESHOLD).to(DEVICE)
    model = CSISpikingNet(
        input_size=X.shape[-1],
        hidden_sizes=HIDDEN_SIZES,
        num_classes=len(mat_loader.ACTIVITY_CODES),
        beta=BETA,
        threshold=LIF_THRESHOLD,
    ).to(DEVICE)

    sample_windows, _ = next(iter(train_loader))
    sample_spikes = encoder(sample_windows.to(DEVICE))
    print(f"spike sparsity at threshold={DELTA_THRESHOLD}: "
          f"{sample_spikes.mean().item():.4f} fraction of entries fire")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\nTraining on {DEVICE} for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, encoder, train_loader, optimizer, loss_fn)
        val_loss, val_acc = evaluate(model, encoder, val_loader, loss_fn)
        print(
            f"epoch {epoch:2d}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  "
            f"({time.time()-t0:.1f}s)"
        )

    test_loss, test_acc = evaluate(model, encoder, test_loader, loss_fn)
    print(f"\ntest_loss={test_loss:.4f} test_acc={test_acc:.3f}")
    print(f"total wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
