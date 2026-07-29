"""Same task, same data, same conv frontend as train_regression.py -- but a
plain GRU instead of the delta-encoded LIF stack, and no spike encoding at
all. Exists purely as a same-conditions comparison point: does the spiking
approach actually do better than a standard, equivalently-sized ANN, or is
the CSI->motion signal similarly extractable either way?

Reuses train_regression's data loading (load_or_build_windows,
make_dataloaders) unchanged, so both scripts train on exactly the same
windows/split (including the same on-disk cache, if one exists).
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from snn_csi_tracking.models.baseline_regressor import CSIConvGRURegressor
from snn_csi_tracking.training.train_regression import (
    BATCH_SIZE,
    CONV_CHANNELS,
    CONV_KERNEL_SIZE,
    DEVICE,
    LEARNING_RATE,
    NUM_EPOCHS,
    load_or_build_windows,
    make_dataloaders,
)

GRU_HIDDEN_SIZE = 64  # roughly comparable parameter count to HIDDEN_SIZES=[128,32] in the SNN
GRU_NUM_LAYERS = 1


def train_one_epoch(model, loader, optimizer, loss_fn) -> float:
    model.train()
    total_loss, total = 0.0, 0
    for windows, labels, _rate_idx in loader:
        windows, labels = windows.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        pred = model(windows).squeeze(-1)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, loss_fn) -> float:
    model.eval()
    total_loss, total = 0.0, 0
    for windows, labels, _rate_idx in loader:
        windows, labels = windows.to(DEVICE), labels.to(DEVICE)
        pred = model(windows).squeeze(-1)
        loss = loss_fn(pred, labels)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / total


def main():
    t_start = time.time()
    print("Loading trials with ground truth available (same data as train_regression.py)...")
    X, y, groups, rates = load_or_build_windows()
    print(f"X={X.shape}, y={y.shape}, {len(set(groups))} trials, "
          f"label range=[{y.min():.2f}, {y.max():.2f}] ({time.time()-t_start:.1f}s)")

    train_loader, val_loader, test_loader = make_dataloaders(X, y, groups, rates, BATCH_SIZE)

    model = CSIConvGRURegressor(
        num_antennas=X.shape[2],
        num_subcarriers=X.shape[3],
        conv_channels=CONV_CHANNELS,
        hidden_size=GRU_HIDDEN_SIZE,
        output_size=1,
        kernel_size=CONV_KERNEL_SIZE,
        num_layers=GRU_NUM_LAYERS,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = None

    print(f"\nTraining on {DEVICE} for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = evaluate(model, val_loader, loss_fn)
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

    test_loss = evaluate(model, test_loader, loss_fn)
    test_label_var = test_loader.dataset.labels.numpy().var()
    print(f"test_mse={test_loss:.4f}  (test-set label variance={test_label_var:.4f}, "
          f"so R^2 = {1 - test_loss / test_label_var:.3f})")
    print(f"total wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
