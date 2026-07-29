"""Reports train/val/test metrics for a saved presence/position checkpoint,
without retraining. Reuses each training script's own evaluate() logic, so
numbers here always match what training reported.

Run:
    .venv/bin/python scripts/evaluate_presence_position.py \
        --checkpoint results/models/presence_position_snn_50ms.pt --kind windowed
    .venv/bin/python scripts/evaluate_presence_position.py \
        --checkpoint results/models/snn_perframe_model_50ms.pt --kind perframe2d
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from snn_csi_tracking.data.dataset import build_datasets
from snn_csi_tracking.data.presence_position_dataset import load_or_build_dataset, load_or_build_perframe_dataset
from snn_csi_tracking.models.presence_position_snn import SNNPresencePosition, SNNPresencePositionPerFrame
from snn_csi_tracking.training import train_presence_position as tw
from snn_csi_tracking.training import train_presence_position_perframe as tp


NUM_SUBCARRIERS = 1024
# channel count -> use_phase, matching presence_position_dataset.compute_features
_CHANNELS_TO_FEATURE_FLAGS = {3: False, 7: True}


def evaluate_windowed(checkpoint_path: str):
    state_dict = torch.load(checkpoint_path, map_location=tw.DEVICE)
    in_features = state_dict["fc1.weight"].shape[1]
    use_phase = _CHANNELS_TO_FEATURE_FLAGS[in_features // NUM_SUBCARRIERS]

    X, y, present, groups, _activity_codes = load_or_build_dataset(
        tw.RAW_ROOT, tw.TRAJECTORY_CACHE_DIR, tw.CACHE_DIR,
        rate_ms=tw.RATE_MS, t_win=tw.T_WIN, stride=tw.STRIDE, use_phase=use_phase,
    )
    train_ds, val_ds, test_ds = build_datasets(X, y, groups, label_dtype=torch.float32, extra=present)

    model = SNNPresencePosition(in_features=in_features, h1=tw.HIDDEN_1, h2=tw.HIDDEN_2).to(tw.DEVICE)
    model.load_state_dict(state_dict)
    bce_loss = nn.BCEWithLogitsLoss()

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        loader = DataLoader(ds, batch_size=tw.BATCH_SIZE, shuffle=False)
        _loss, acc, pos_err = tw.evaluate(model, loader, bce_loss)
        print(f"{name:5s}: presence_acc={acc:.3f}  position_error={pos_err:.4f}  n={len(ds)}")


def evaluate_perframe(checkpoint_path: str, out_dim: int):
    use_3d = out_dim == 3
    state_dict = torch.load(checkpoint_path, map_location=tp.DEVICE)
    in_features = state_dict["fc1.weight"].shape[1]
    use_phase = _CHANNELS_TO_FEATURE_FLAGS[in_features // NUM_SUBCARRIERS]

    X, pos, present, groups, _activity_codes = load_or_build_perframe_dataset(
        tp.RAW_ROOT, tp.TRAJECTORY_CACHE_DIR, tp.DEPTH_CACHE_DIR, tp.CACHE_DIR,
        rate_ms=tp.RATE_MS, t_win=tp.T_WIN, stride=tp.STRIDE, use_3d=use_3d, use_phase=use_phase,
    )
    train_ds, val_ds, test_ds = build_datasets(X, pos, groups, label_dtype=torch.float32, extra=present)

    model = SNNPresencePositionPerFrame(
        in_features=in_features, h1=tp.HIDDEN_1, h2=tp.HIDDEN_2, out_dim=out_dim
    ).to(tp.DEVICE)
    model.load_state_dict(state_dict)
    bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        loader = DataLoader(ds, batch_size=tp.BATCH_SIZE, shuffle=False)
        _loss, acc, pos_err = tp.evaluate(model, loader, bce_loss)
        print(f"{name:5s}: presence_acc={acc:.3f}  position_error={pos_err:.4f}  n={len(ds)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kind", choices=["windowed", "perframe2d", "perframe3d"], required=True)
    args = parser.parse_args()

    print(f"Evaluating {args.checkpoint} ({args.kind})...")
    if args.kind == "windowed":
        evaluate_windowed(args.checkpoint)
    elif args.kind == "perframe2d":
        evaluate_perframe(args.checkpoint, out_dim=2)
    else:
        evaluate_perframe(args.checkpoint, out_dim=3)


if __name__ == "__main__":
    main()
