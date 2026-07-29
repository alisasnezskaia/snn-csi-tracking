"""Energy estimate for THE joint presence+position tracking configuration
(training.train_presence_position_conv, --model snn vs --model ann) --
matches the paper abstract's actual claim, not the scalar-regression task
(see estimate_energy.py for that one, which this script's methodology
mirrors).

Same standard methodology (Horowitz, "Computing's Energy Problem", ISSCC
2014):
    E_MAC ~= 4.6 pJ   (dense multiply-accumulate -- every ANN synapse,
                        every timestep, regardless of value)
    E_AC  ~= 0.9 pJ    (accumulate-only -- an SNN synapse, ONLY when its
                        presynaptic neuron actually spikes)

Unlike estimate_energy.py's synthetic random batch, this replays the ACTUAL
trained checkpoints against the REAL held-out test split (same session-
grouped BALANCED_SPLIT used at training time) -- spike rates measured on
real CSI, not noise.

Per-layer accounting for SNNPresencePositionConvPerFrame:
  fc1          (conv_features -> h1):  input = delta-encoded spikes -- gated
  fc2          (h1 -> h2):              input = spk1                -- gated
  fc_presence  (h2 -> 1):               input = spk2                -- gated
  fc_position.0(h2 -> 16):              input = spk2                -- gated
  fc_position.2(16 -> out_dim):         input = ReLU(...), continuous
                                         -- NOT spike-gated in either model
                                         (a small dense readout off the
                                         spiking trunk, same convention as
                                         CSIConvSpikingRegressor's readout)
The shared PerFrameConvEncoder frontend is identical (dense) in both models
and excluded from the comparison, matching estimate_energy.py's convention
-- the energy story here is specifically about the spiking-vs-dense
fully-connected trunk, not the conv frontend both models share as-is.

Run (after training both --model snn and --model ann checkpoints):
    .venv/bin/python scripts/estimate_energy_presence_position.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_csi_tracking.data.presence_position_dataset import load_or_build_perframe_dataset
from snn_csi_tracking.models.presence_position_ann import ANNPresencePositionConvPerFrame
from snn_csi_tracking.models.presence_position_snn import SNNPresencePositionConvPerFrame
from snn_csi_tracking.training.train_presence_position_conv import (
    AMPLITUDE_NORM, BATCH_SIZE, CACHE_DIR, CONV_CHANNELS, DELTA_THRESHOLD, DEPTH_CACHE_DIR, HIDDEN_1, HIDDEN_2,
    KERNEL_SIZE, RATE_MS, RAW_ROOT, STRIDE, T_WIN, TRAJECTORY_CACHE_DIR, USE_EMPTY_BASELINE,
    USE_MOTION_MAGNITUDE, USE_PHASE, USE_RELATIVE_MOTION, make_dataloaders,
)

E_MAC_PJ = 4.6
E_AC_PJ = 0.9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def snn_layer_spike_rates(model: SNNPresencePositionConvPerFrame, windows: torch.Tensor) -> dict[str, float]:
    """Replays SNNPresencePositionConvPerFrame.forward, recording the mean
    firing rate of every spiking layer's own input. Returns a dict keyed by
    layer name (see module docstring's per-layer accounting)."""
    t = windows.shape[-1]
    conv_in = windows.permute(0, 3, 1, 2)
    features = model.conv_encoder(conv_in)
    diff = torch.diff(features, dim=1, prepend=features[:, :1])
    spikes = (diff >= model.delta_threshold).float()  # input to fc1, every step

    mem1, mem2 = model.lif1.init_leaky(), model.lif2.init_leaky()
    sums = {"fc1": 0.0, "fc2": 0.0, "fc3": 0.0}  # fc3 = shared input to fc_presence & fc_position.0
    counts = {"fc1": 0, "fc2": 0, "fc3": 0}
    for step in range(t):
        cur = spikes[:, step]
        sums["fc1"] += cur.sum().item(); counts["fc1"] += cur.numel()
        spk1, mem1 = model.lif1(model.fc1(cur), mem1)
        sums["fc2"] += spk1.sum().item(); counts["fc2"] += spk1.numel()
        spk2, mem2 = model.lif2(model.fc2(spk1), mem2)
        sums["fc3"] += spk2.sum().item(); counts["fc3"] += spk2.numel()
    return {k: sums[k] / counts[k] for k in sums}


def layer_specs(model) -> list[tuple[str, int, int, str]]:
    """[(name, in_features, out_features, rate_key_or_None), ...] --
    rate_key_or_None is None for the always-dense fc_position.2 readout."""
    h2 = model.fc2.out_features
    return [
        ("fc1", model.fc1.in_features, model.fc1.out_features, "fc1"),
        ("fc2", model.fc2.in_features, model.fc2.out_features, "fc2"),
        ("fc_presence", model.fc_presence.in_features, model.fc_presence.out_features, "fc3"),
        ("fc_position.0", h2, model.fc_position[0].out_features, "fc3"),
        ("fc_position.2", model.fc_position[2].in_features, model.fc_position[2].out_features, None),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-3d", action="store_true", help="compare the 3D (x,y,z) checkpoints instead of 2D")
    args = parser.parse_args()
    out_dim = 3 if args.use_3d else 2
    dim_tag = "3d" if args.use_3d else "2d"

    print(f"Loading {RATE_MS}ms captures (same config used to train presence_position_conv models: "
          f"use_3d={args.use_3d}, use_empty_baseline={USE_EMPTY_BASELINE}, use_motion_magnitude={USE_MOTION_MAGNITUDE}, "
          f"use_relative_motion={USE_RELATIVE_MOTION})...")
    X, pos, present, groups, activity_codes = load_or_build_perframe_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, DEPTH_CACHE_DIR, CACHE_DIR,
        rate_ms=RATE_MS, t_win=T_WIN, stride=STRIDE, use_3d=args.use_3d, use_phase=USE_PHASE,
        use_empty_baseline=USE_EMPTY_BASELINE, denoise=None, use_motion_magnitude=USE_MOTION_MAGNITUDE,
        amplitude_norm=AMPLITUDE_NORM, use_relative_motion=USE_RELATIVE_MOTION,
    )
    _train_loader, _val_loader, test_loader, _train_ds, _test_ds = make_dataloaders(
        X, pos, present, groups, activity_codes, BATCH_SIZE, split_seed=0, split_mode="balanced"
    )

    num_channels, num_subcarriers = X.shape[1], X.shape[2]
    snn_model = SNNPresencePositionConvPerFrame(
        num_channels=num_channels, num_subcarriers=num_subcarriers, conv_channels=CONV_CHANNELS,
        h1=HIDDEN_1, h2=HIDDEN_2, out_dim=out_dim, kernel_size=KERNEL_SIZE, delta_threshold=DELTA_THRESHOLD,
    ).to(DEVICE)
    ann_model = ANNPresencePositionConvPerFrame(
        num_channels=num_channels, num_subcarriers=num_subcarriers, conv_channels=CONV_CHANNELS,
        h1=HIDDEN_1, h2=HIDDEN_2, out_dim=out_dim, kernel_size=KERNEL_SIZE,
    ).to(DEVICE)

    snn_ckpt = REPO_ROOT / "results" / "models" / f"presence_position_conv_snn_{dim_tag}_none_50ms.pt"
    ann_ckpt = REPO_ROOT / "results" / "models" / f"presence_position_conv_ann_{dim_tag}_none_50ms.pt"
    for name, model, ckpt in [("SNN", snn_model, snn_ckpt), ("ANN", ann_model, ann_ckpt)]:
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            print(f"loaded {name} checkpoint from {ckpt}")
        else:
            print(f"WARNING: no trained {name} checkpoint at {ckpt} -- using randomly-initialized weights, "
                  f"spike rates will NOT be representative")
    snn_model.eval()
    ann_model.eval()

    print("Measuring per-layer spike rates on the real held-out test split...")
    rate_sums = {"fc1": 0.0, "fc2": 0.0, "fc3": 0.0}
    rate_batches = 0
    n_windows = 0
    for windows, _pos_seq, _pres_seq in test_loader:
        windows = windows.to(DEVICE)
        rates = snn_layer_spike_rates(snn_model, windows)
        for k in rate_sums:
            rate_sums[k] += rates[k]
        rate_batches += 1
        n_windows += windows.shape[0]
    spike_rates = {k: v / rate_batches for k, v in rate_sums.items()}
    print(f"measured over {n_windows} test windows ({rate_batches} batches)")
    print(f"per-layer input spike rates: { {k: round(v, 4) for k, v in spike_rates.items()} }")

    specs = layer_specs(snn_model)
    total_snn_pj, total_ann_pj = 0.0, 0.0
    print(f"\nPer-layer energy (T={T_WIN} timesteps), per example:")
    for name, in_f, out_f, rate_key in specs:
        ops_per_step = in_f * out_f
        snn_rate = spike_rates[rate_key] if rate_key is not None else 1.0
        snn_pj = ops_per_step * T_WIN * E_AC_PJ * snn_rate
        ann_pj = ops_per_step * T_WIN * E_MAC_PJ  # always dense
        total_snn_pj += snn_pj
        total_ann_pj += ann_pj
        gated = f"rate={snn_rate:.4f}" if rate_key is not None else "dense (unGated, both models)"
        print(f"  {name:16s} ({in_f:4d}x{out_f:3d}): SNN={snn_pj/1e3:8.2f} nJ  ANN={ann_pj/1e3:8.2f} nJ  [{gated}]")

    print()
    print(f"Estimated energy per window (T={T_WIN} timesteps @ {RATE_MS}ms = {T_WIN*RATE_MS/1000:.1f}s of CSI), per example:")
    print(f"  SNN (spike-gated AC, {E_AC_PJ} pJ/op):  {total_snn_pj/1e3:.2f} nJ")
    print(f"  ANN (dense MAC, {E_MAC_PJ} pJ/op):       {total_ann_pj/1e3:.2f} nJ")
    print(f"  ANN / SNN ratio: {total_ann_pj/total_snn_pj:.1f}x more energy for the dense ANN")


if __name__ == "__main__":
    main()
