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

Same convention as estimate_energy_presence_position.py: reports BOTH TRUNK
ONLY (recurrent trunk + heads, encoder excluded) and FULL PIPELINE (+ the
shared PerFrameConvEncoder frontend -- identical, dense, in all three
models). --feature must match whichever config informs the SNN/ANN
reference numbers below (see train_presence_position_conv.py's
FEATURE_KWARGS) -- LSTM's own number needs no checkpoint (weight-independent,
architecture-shape-only), but the encoder's channel count does depend on it.

Run:
    .venv/bin/python scripts/estimate_energy_lstm.py --feature cross_coherence
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
    AMPLITUDE_NORM, CACHE_DIR, CONV_CHANNELS, DEPTH_CACHE_DIR, FEATURE_KWARGS, HIDDEN_1, HIDDEN_2, KERNEL_SIZE,
    RATE_MS, RAW_ROOT, STRIDE, T_WIN, TRAJECTORY_CACHE_DIR, USE_EMPTY_BASELINE, USE_PHASE,
)

E_MAC_PJ = 4.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def lstm_macs_per_step(lstm: torch.nn.LSTM) -> int:
    """4 gates (input/forget/cell/output), each one (in->hidden) + one
    (hidden->hidden) dense matmul."""
    in_f, hidden = lstm.input_size, lstm.hidden_size
    return 4 * (in_f * hidden + hidden * hidden)


def conv_encoder_macs_per_frame(conv_encoder, num_channels: int, num_subcarriers: int) -> int:
    """Same as estimate_energy_presence_position.py's helper of the same name
    -- duplicated here rather than imported, matching this script's existing
    standalone-per-script convention (see estimate_energy_presence_position.py
    for the formula's derivation/hand-verification)."""
    macs = 0

    def hook(module, _inp, out):
        nonlocal macs
        macs += module.out_channels * module.in_channels * module.kernel_size[0] * out.shape[-1]

    handles = [m.register_forward_hook(hook) for m in conv_encoder.conv if isinstance(m, torch.nn.Conv1d)]
    with torch.no_grad():
        conv_encoder.conv(torch.zeros(1, num_channels, num_subcarriers, device=next(conv_encoder.parameters()).device))
    for h in handles:
        h.remove()
    return macs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-3d", action="store_true", help="match the 3D (x,y,z) checkpoints instead of 2D")
    parser.add_argument("--no-pinhole", dest="pinhole", action="store_false", default=True,
                         help="only meaningful with --use-3d: match the --no-pinhole (normalized [0,1], 'dim_tag=3dz') "
                              "dataset/cache instead of the real-meters pinhole one -- MUST match whichever scale "
                              "the SNN/ANN reference numbers below came from, same convention as "
                              "estimate_energy_presence_position.py.")
    parser.add_argument("--feature", choices=list(FEATURE_KWARGS), default="cross_coherence",
                         help="picks the dataset's channel count (and thus the encoder's MAC count) -- "
                              "should match whatever config the SNN/ANN reference numbers below came from.")
    args = parser.parse_args()
    out_dim = 3 if args.use_3d else 2

    print(f"Loading dataset just for shape info (amplitude_norm={AMPLITUDE_NORM}, use_3d={args.use_3d}, "
          f"pinhole={args.pinhole}, feature={args.feature} ({FEATURE_KWARGS[args.feature]})) -- matches "
          f"estimate_energy_presence_position.py's config...")
    X, _pos, _present, _groups, _activity_codes = load_or_build_perframe_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, DEPTH_CACHE_DIR, CACHE_DIR,
        rate_ms=RATE_MS, t_win=T_WIN, stride=STRIDE, use_3d=args.use_3d, pinhole=args.pinhole, use_phase=USE_PHASE,
        use_empty_baseline=USE_EMPTY_BASELINE, denoise=None, amplitude_norm=AMPLITUDE_NORM,
        **FEATURE_KWARGS[args.feature],
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

    # Reference SNN/ANN numbers -- hardcoded, not measured here, since that script is the
    # one that replays real checkpoints for spike rates. STALE as of this edit (predate
    # the --feature fix -- see estimate_energy_presence_position.py's docstring on the
    # checkpoint/dataset-config mismatch this used to silently allow). Re-run that
    # script with --feature matching this run and update these before trusting the
    # ratio below -- these are a placeholder, not a live measurement.
    snn_reference_nj, ann_reference_nj = (9477.34, 310080.92) if args.use_3d else (131.35, 310076.21)

    encoder_macs = conv_encoder_macs_per_frame(model.conv_encoder, num_channels, num_subcarriers)
    encoder_pj = encoder_macs * T_WIN * E_MAC_PJ

    print(f"\nEstimated energy per window (T={T_WIN} timesteps @ {RATE_MS}ms = {T_WIN*RATE_MS/1000:.1f}s of CSI), per example:")
    print(f"  -- TRUNK ONLY (encoder excluded) --")
    print(f"  LSTM (dense, {E_MAC_PJ} pJ/op, no sparsity): {total_pj/1e3:.2f} nJ")
    print(f"  (compare against estimate_energy_presence_position.py --feature {args.feature} --use-3d={args.use_3d}: "
          f"SNN={snn_reference_nj} nJ, ANN={ann_reference_nj} nJ -- STALE placeholder, see comment above)")
    print(f"  LSTM / SNN ratio: {total_pj/1e3/snn_reference_nj:.1f}x more energy than the SNN")
    print(f"\n  -- FULL PIPELINE (shared dense encoder + trunk) --")
    print(f"  PerFrameConvEncoder ({encoder_macs} MACs/frame, always dense, identical in every model): {encoder_pj/1e3:.2f} nJ/window")
    total_lstm_full_pj = total_pj + encoder_pj
    print(f"  LSTM total: {total_lstm_full_pj/1e3:.2f} nJ  (encoder is {encoder_pj/total_lstm_full_pj*100:.1f}% of it)")
    print(f"  SNN/ANN full-pipeline totals: rerun estimate_energy_presence_position.py --feature {args.feature} "
          f"for the real (encoder+trunk) SNN/ANN numbers to compare against -- the hardcoded references above "
          f"are trunk-only and can't be validly added to here.")


if __name__ == "__main__":
    main()
