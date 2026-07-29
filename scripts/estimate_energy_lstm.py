"""Energy estimate for LSTMPresencePositionConvPerFrame, using the same
Horowitz (ISSCC 2014) methodology as estimate_energy_presence_position.py
(E_MAC ~= 4.6 pJ per dense multiply-accumulate) -- extends that SNN-vs-ANN
comparison to a third point: a proper conventional recurrent benchmark.

Unlike the SNN, the LSTM has NO sparsity to measure: every gate computes
for every unit at every timestep regardless of input values or weights.
That means, unlike estimate_energy_presence_position.py (which MUST replay
real trained checkpoints against real held-out data to get a real spike
rate), this number is fully determined by architecture shape alone -- no
trained checkpoint or test data needed. An nn.LSTM layer has 4 gates
(input, forget, cell, output), each requiring one (in_features -> hidden)
and one (hidden -> hidden) dense matmul, so MACs/timestep for one LSTM
layer = 4 * (in_features*hidden + hidden*hidden).

Same convention as estimate_energy_presence_position.py: the shared
PerFrameConvEncoder frontend is excluded from the comparison (identical,
dense, in all three models) -- this is specifically about the recurrent
trunk + heads.

Run:
    .venv/bin/python scripts/estimate_energy_lstm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import argparse

import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_csi_tracking.data.presence_position_dataset import load_or_build_perframe_dataset
from snn_csi_tracking.models.presence_position_lstm import LSTMPresencePositionConvPerFrame
from snn_csi_tracking.training.train_presence_position_conv import (
    AMPLITUDE_NORM, CACHE_DIR, CONV_CHANNELS, DEPTH_CACHE_DIR, HIDDEN_1, HIDDEN_2, KERNEL_SIZE,
    RATE_MS, RAW_ROOT, STRIDE, T_WIN, TRAJECTORY_CACHE_DIR, USE_EMPTY_BASELINE, USE_MOTION_MAGNITUDE,
    USE_PHASE, USE_RELATIVE_MOTION,
)

E_MAC_PJ = 4.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def lstm_macs_per_step(lstm: torch.nn.LSTM) -> int:
    """4 gates (input/forget/cell/output), each one (in->hidden) + one
    (hidden->hidden) dense matmul."""
    in_f, hidden = lstm.input_size, lstm.hidden_size
    return 4 * (in_f * hidden + hidden * hidden)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-3d", action="store_true", help="match the 3D (x,y,z) checkpoints instead of 2D")
    args = parser.parse_args()
    out_dim = 3 if args.use_3d else 2

    print(f"Loading dataset just for shape info (amplitude_norm={AMPLITUDE_NORM}, use_3d={args.use_3d}, "
          f"use_relative_motion={USE_RELATIVE_MOTION}) -- matches estimate_energy_presence_position.py's config...")
    X, _pos, _present, _groups, _activity_codes = load_or_build_perframe_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, DEPTH_CACHE_DIR, CACHE_DIR,
        rate_ms=RATE_MS, t_win=T_WIN, stride=STRIDE, use_3d=args.use_3d, use_phase=USE_PHASE,
        use_empty_baseline=USE_EMPTY_BASELINE, denoise=None, use_motion_magnitude=USE_MOTION_MAGNITUDE,
        amplitude_norm=AMPLITUDE_NORM, use_relative_motion=USE_RELATIVE_MOTION,
    )
    num_channels, num_subcarriers = X.shape[1], X.shape[2]

    model = LSTMPresencePositionConvPerFrame(
        num_channels=num_channels, num_subcarriers=num_subcarriers, conv_channels=CONV_CHANNELS,
        h1=HIDDEN_1, h2=HIDDEN_2, out_dim=out_dim, kernel_size=KERNEL_SIZE,
    ).to(DEVICE)

    specs = [
        ("lstm  (conv_features -> h1)", lstm_macs_per_step(model.lstm)),
        ("lstm2 (h1 -> h2)", lstm_macs_per_step(model.lstm2)),
        ("fc_presence (h2 -> 1)", model.fc_presence.in_features * model.fc_presence.out_features),
        ("fc_position.0 (h2 -> 16)", model.fc_position[0].in_features * model.fc_position[0].out_features),
        ("fc_position.2 (16 -> out_dim)", model.fc_position[2].in_features * model.fc_position[2].out_features),
    ]

    print(f"\nPer-layer energy (T={T_WIN} timesteps, all dense -- no sparsity to gate), per example:")
    total_pj = 0.0
    for name, macs in specs:
        pj = macs * T_WIN * E_MAC_PJ
        total_pj += pj
        print(f"  {name:32s}: {pj/1e3:8.2f} nJ  ({macs} MACs/step)")

    # Reference SNN/ANN numbers from estimate_energy_presence_position.py's last run --
    # hardcoded, not measured here, since that script is the one that replays real
    # checkpoints for spike rates. Re-run that script and update these if the SNN/ANN
    # checkpoints change; stale numbers here would silently mis-report the ratio below.
    snn_reference_nj, ann_reference_nj = (9477.34, 310080.92) if args.use_3d else (131.35, 310076.21)

    print(f"\nEstimated energy per window (T={T_WIN} timesteps @ {RATE_MS}ms = {T_WIN*RATE_MS/1000:.1f}s of CSI), per example:")
    print(f"  LSTM (dense, {E_MAC_PJ} pJ/op, no sparsity): {total_pj/1e3:.2f} nJ")
    print(f"  (compare against estimate_energy_presence_position.py --use-3d={args.use_3d}: "
          f"SNN={snn_reference_nj} nJ, ANN={ann_reference_nj} nJ)")
    print(f"  LSTM / SNN ratio: {total_pj/1e3/snn_reference_nj:.1f}x more energy than the SNN")


if __name__ == "__main__":
    main()
