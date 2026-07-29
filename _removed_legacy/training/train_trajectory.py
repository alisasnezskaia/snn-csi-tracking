"""Full position-trajectory regression attempt -- predicting average (x, y)
pixel position per window directly, instead of the motion/speed proxy in
train_regression.py.

Run separately, expected to underperform: see conversation notes on why a
single 3-antenna link + uncalibrated, occasionally-gappy camera labels
shouldn't reliably support absolute position. This exists specifically to
turn that argument into a measured result -- a documented negative result
here is what justifies having used speed_labels as the real target, rather
than just asserting it.

Same data, same conv+SNN architecture, same split as train_regression.py --
only the label (position_labels instead of speed_labels) and the output
size (2 instead of 1) differ.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from snn_csi_tracking.data import mat_loader
from snn_csi_tracking.data.dataset import build_datasets
from snn_csi_tracking.data.preprocessing import normalize, position_labels, sliding_windows
from snn_csi_tracking.models.encoding import DeltaEncoder
from snn_csi_tracking.models.regression_snn import CSIConvSpikingRegressor
from snn_csi_tracking.training.train_regression import (
    BATCH_SIZE,
    BETA,
    CONDITIONS,
    CONV_CHANNELS,
    CONV_KERNEL_SIZE,
    DATA_ROOT,
    DELTA_THRESHOLD,
    DEVICE,
    HIDDEN_SIZES,
    LEARNING_RATE,
    LIF_THRESHOLD,
    NUM_EPOCHS,
    RATE_MS,
    STRIDE,
    TRAJECTORY_DIR,
    WINDOW_SIZE,
)


def build_windows_and_position_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same as train_regression.build_windows_and_labels, but y is (NumWindows, 2)
    average (x_px, y_px) per window instead of a single speed scalar."""
    X, y, groups = [], [], []
    trial_id = 0
    n_skipped_no_gt = 0
    for condition in CONDITIONS:
        path = DATA_ROOT / condition / f"csi_office_{RATE_MS}ms_interframe.mat"
        if not path.exists():
            continue
        for row in mat_loader.load_table_metadata(path):
            code = row["activity_code"]
            traj_path = TRAJECTORY_DIR / f"{condition}_{RATE_MS}ms_{code}_{row['filename']}.npy"
            if not traj_path.exists():
                n_skipped_no_gt += 1
                continue
            csi = mat_loader.load_csi(path, row["csi_key"], row["row_index"])
            if csi is None:
                continue
            positions = np.load(traj_path)
            if len(positions) != csi.shape[0]:
                print(f"  WARNING: {traj_path.name} has {len(positions)} positions "
                      f"but csi has {csi.shape[0]} frames -- skipping")
                continue

            amp = normalize(csi)
            amp = amp.transpose(0, 2, 1)  # (T, NumAntennas, NumSubcarriers)
            windows = sliding_windows(amp, WINDOW_SIZE, STRIDE)
            labels = position_labels(positions, WINDOW_SIZE, STRIDE)  # (NumWin, 2)

            X.append(windows)
            y.append(labels)
            groups.extend([trial_id] * windows.shape[0])
            trial_id += 1

    print(f"{trial_id} trials used, {n_skipped_no_gt} skipped (no ground truth yet)")
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0), np.array(groups)


def make_dataloaders(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = build_datasets(X, y, groups, label_dtype=torch.float32)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def train_one_epoch(model, encoder, loader, optimizer, loss_fn) -> float:
    model.train()
    total_loss, total = 0.0, 0
    for windows, labels in loader:
        windows, labels = windows.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        pred = model(windows, encoder)  # (batch, 2)
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
    for windows, labels in loader:
        windows, labels = windows.to(DEVICE), labels.to(DEVICE)
        pred = model(windows, encoder)
        loss = loss_fn(pred, labels)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / total


def main():
    t_start = time.time()
    print(f"Loading {RATE_MS}ms trials with ground truth available...")
    X, y, groups = build_windows_and_position_labels()
    print(f"X={X.shape}, y={y.shape}, {len(set(groups))} trials, "
          f"x range=[{y[:,0].min():.1f}, {y[:,0].max():.1f}], "
          f"y range=[{y[:,1].min():.1f}, {y[:,1].max():.1f}] ({time.time()-t_start:.1f}s)")

    train_loader, val_loader, test_loader = make_dataloaders(X, y, groups, BATCH_SIZE)

    encoder = DeltaEncoder(threshold=DELTA_THRESHOLD).to(DEVICE)
    model = CSIConvSpikingRegressor(
        num_antennas=X.shape[2],
        num_subcarriers=X.shape[3],
        conv_channels=CONV_CHANNELS,
        hidden_sizes=HIDDEN_SIZES,
        output_size=2,  # (x, y) instead of a single speed value
        beta=BETA,
        threshold=LIF_THRESHOLD,
        kernel_size=CONV_KERNEL_SIZE,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
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

    print(f"\nrestoring best checkpoint from epoch {best_epoch} (val_mse={best_val_loss:.4f})")
    model.load_state_dict(best_state)

    test_loss = evaluate(model, encoder, test_loader, loss_fn)
    total_variance = y.var(axis=0).sum()  # combined x+y variance
    print(f"test_mse={test_loss:.4f}  (combined label variance={total_variance:.4f}, "
          f"so R^2 ~ {1 - test_loss / total_variance:.3f})")
    print(f"total wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
