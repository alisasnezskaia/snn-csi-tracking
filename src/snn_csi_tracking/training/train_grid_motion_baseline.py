"""Non-spiking baseline for the grid-motion task: identical conv frontend,
data, held-out-trial split, log1p label transform, and motion-weighted
training sampler as train_grid_motion.py, but a GRU instead of a
delta-encoded LIF stack.

Exists to answer "is the weak grid-motion result (test_r2 around 0) coming
from the SNN-specific pieces, or from a real data/hardware limit regardless
of architecture" -- two concrete candidate SNN-specific issues this
sidesteps:

  1. DeltaEncoder only passes through features that CHANGE frame-to-frame --
     well-suited to "how much motion happened" but potentially wrong for
     location, whose signature may live in an absolute pattern (e.g. which
     phase-difference channels sit at a particular level) rather than in
     when things change. A GRU sees the raw continuous conv features
     directly, no delta encoding.
  2. The LIF readout's effective memory is short (~10 timesteps at
     beta=0.9) versus the label aggregating the full 100-timestep/5s
     window. A GRU's hidden state isn't bound to that same decay.

This is the same comparison baseline_regressor.py already makes for the
scalar motion-regression task ("did the spiking part earn its keep"),
extended to the grid task.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from snn_csi_tracking.models.baseline_regressor import CSIConvGRURegressor
from snn_csi_tracking.training.train_grid_motion import (
    BATCH_SIZE,
    CONV_CHANNELS,
    CONV_KERNEL_SIZE,
    DEVICE,
    GRID_DIM,
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    load_or_build_dataset,
    make_dataloaders,
)

GRU_HIDDEN_SIZE = 128


def train_one_epoch(model, loader, optimizer, loss_fn) -> float:
    model.train()
    total_loss, total = 0.0, 0
    for windows, labels in loader:
        windows, labels = windows.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        pred = model(windows)
        loss = loss_fn(pred, torch.log1p(labels))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader) -> tuple[float, float]:
    """Returns (raw-space mse, raw-space r2) -- same convention as
    train_grid_motion.evaluate(), for a directly comparable number."""
    model.eval()
    all_pred, all_true = [], []
    for windows, labels in loader:
        windows, labels = windows.to(DEVICE), labels.to(DEVICE)
        pred = model(windows)
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
    print(f"Loading cached grid-motion dataset ({GRID_DIM}x{GRID_DIM} grid)...")
    X, y, groups, activity_codes = load_or_build_dataset()
    print(f"X={X.shape}, y={y.shape}, {len(set(groups))} trials ({time.time()-t_start:.1f}s)")

    train_loader, val_loader, test_loader = make_dataloaders(X, y, groups, activity_codes, BATCH_SIZE)

    k = GRID_DIM * GRID_DIM
    model = CSIConvGRURegressor(
        num_antennas=X.shape[2],
        num_subcarriers=X.shape[3],
        conv_channels=CONV_CHANNELS,
        hidden_size=GRU_HIDDEN_SIZE,
        output_size=k,
        kernel_size=CONV_KERNEL_SIZE,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = None

    print(f"\nTraining GRU (non-spiking) baseline on {DEVICE} for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_r2 = evaluate(model, val_loader)
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
            best_epoch = epoch
            marker = "  <- best so far"
        print(f"epoch {epoch:3d}/{NUM_EPOCHS}  train_mse={train_loss:.4f}  "
              f"val_mse={val_loss:.4f}  val_r2={val_r2:.3f}  ({time.time()-t0:.1f}s){marker}")

    print(f"\nrestoring best checkpoint from epoch {best_epoch} (val_mse={best_val_loss:.4f})")
    model.load_state_dict(best_state)

    test_mse, test_r2 = evaluate(model, test_loader)
    print(f"test_mse={test_mse:.4f}  test_r2={test_r2:.3f}")
    print(f"total wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
