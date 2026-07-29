"""Energy estimate: SNN (models.regression_snn.CSIConvSpikingRegressor,
spike-gated accumulate/AC operations) vs. its matched ANN twin
(models.regression_ann.CSIConvANNRegressor, dense multiply-accumulate/MAC
operations every timestep), for the paper's energy-efficiency claim.

Standard methodology (see e.g. Horowitz, "Computing's Energy Problem (and
what we can do about it)", ISSCC 2014 -- the commonly-cited 45nm CMOS
figures used across the SNN literature for exactly this comparison):
    E_MAC ~= 4.6 pJ   (32-bit float multiply-accumulate -- what every ANN
                        synapse does, every timestep, regardless of value)
    E_AC  ~= 0.9 pJ    (accumulate only, no multiply -- what an SNN synapse
                        does, and ONLY when its presynaptic neuron actually
                        spikes; a silent synapse (input=0) contributes
                        nothing on real spiking hardware)

For a layer with (in_features x out_features) synapses running for T
timesteps over a batch:
    ANN energy = in_features * out_features * T * E_MAC            (always)
    SNN energy = in_features * out_features * T * E_AC * spike_rate (gated)

spike_rate is measured directly from the trained model's actual behavior on
real held-out data (per layer -- the delta-encoder's own sparsity is NOT
assumed to equal every hidden layer's spike rate, since each LIF layer's
firing rate is its own function of its trained weights).

Run:
    .venv/bin/python scripts/estimate_energy.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from snn_csi_tracking.models.encoding import DeltaEncoder
from snn_csi_tracking.models.regression_ann import CSIConvANNRegressor
from snn_csi_tracking.models.regression_snn import CSIConvSpikingRegressor

E_MAC_PJ = 4.6  # dense multiply-accumulate, 32-bit float, 45nm CMOS (Horowitz 2014)
E_AC_PJ = 0.9   # accumulate-only (spike-gated), same source

REPO_ROOT = Path(__file__).parent.parent
CACHE_DIR = REPO_ROOT / "data" / "processed"

CONV_CHANNELS = [16, 32]
CONV_KERNEL_SIZE = 9
HIDDEN_SIZES = [128, 32]
RATE_EMBED_DIM = 8
NUM_RATES = 4
BETA = 0.9
LIF_THRESHOLD = 1.0
DELTA_THRESHOLD = 0.3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def snn_layer_spike_rates(model: CSIConvSpikingRegressor, encoder: DeltaEncoder, csi: torch.Tensor, rate_idx: torch.Tensor) -> list[float]:
    """Replays CSIConvSpikingRegressor.forward, but also records the mean
    firing rate of every hidden layer's spike output (not just the delta
    encoder's input sparsity) -- each LIF layer's own rate depends on its
    trained weights, not just the input it's given.
    Returns [rate_into_hidden0, rate_into_hidden1, ..., rate_into_readout]
    -- one entry per synaptic layer (len(hidden_layers) + 1), each the
    firing rate of the layer's OWN input.
    """
    features = model.conv_encoder(csi)
    spikes = encoder(features)  # (batch, T, feat) -- input to hidden layer 0

    batch_size, num_steps, _ = spikes.shape
    hidden_mem = [lif.init_leaky() for lif in model.hidden_lifs]
    readout_mem = model.readout_lif.init_leaky()
    rate_vec = model.rate_embedding(rate_idx) if model.rate_embedding is not None else None

    input_sums = [0.0] * (len(model.hidden_layers) + 1)
    input_counts = [0] * (len(model.hidden_layers) + 1)

    for t in range(num_steps):
        cur = spikes[:, t, :]
        if rate_vec is not None:
            cur = torch.cat([cur, rate_vec], dim=-1)
        for i, (layer, lif) in enumerate(zip(model.hidden_layers, model.hidden_lifs)):
            input_sums[i] += cur.mean().item() * cur.numel()
            input_counts[i] += cur.numel()
            cur, hidden_mem[i] = lif(layer(cur), hidden_mem[i])
        input_sums[-1] += cur.mean().item() * cur.numel()
        input_counts[-1] += cur.numel()
        _, readout_mem = model.readout_lif(model.readout_layer(cur), readout_mem)

    return [s / c for s, c in zip(input_sums, input_counts)]


def layer_sizes(model: CSIConvSpikingRegressor | CSIConvANNRegressor) -> list[tuple[int, int]]:
    """[(in_features, out_features), ...] for every synaptic (Linear) layer,
    hidden layers then readout, in forward order."""
    sizes = []
    for layer in model.hidden_layers:
        sizes.append((layer.in_features, layer.out_features))
    sizes.append((model.readout_layer.in_features, model.readout_layer.out_features))
    return sizes


def main():
    print("Building small synthetic-but-realistic batch (same shapes as real windows)...")
    # Shapes matching train_regression.py's config: (batch, T, chan, sub).
    num_channels = 7  # amp+phase, see preprocessing.extract_features / num_feature_channels(3 antennas)
    num_subcarriers = 1024
    t_win = 128
    batch = 32

    torch.manual_seed(0)
    csi = torch.randn(batch, t_win, num_channels, num_subcarriers).to(DEVICE)
    rate_idx = torch.randint(0, NUM_RATES, (batch,)).to(DEVICE)

    encoder = DeltaEncoder(threshold=DELTA_THRESHOLD).to(DEVICE)
    snn_model = CSIConvSpikingRegressor(
        num_antennas=num_channels, num_subcarriers=num_subcarriers, conv_channels=CONV_CHANNELS,
        hidden_sizes=HIDDEN_SIZES, output_size=1, beta=BETA, threshold=LIF_THRESHOLD,
        kernel_size=CONV_KERNEL_SIZE, num_rates=NUM_RATES, rate_embed_dim=RATE_EMBED_DIM,
    ).to(DEVICE)

    ckpt_path = REPO_ROOT / "results" / "models" / "regression_snn.pt"
    if ckpt_path.exists():
        print(f"Loading trained checkpoint from {ckpt_path}")
        snn_model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    else:
        print(f"WARNING: no trained checkpoint at {ckpt_path} -- using randomly-initialized weights. "
              f"Spike rates from an untrained model are NOT representative of the real energy profile; "
              f"re-run this script after training finishes and the checkpoint is saved.")
    snn_model.eval()

    print("Measuring per-layer spike rates on this batch...")
    spike_rates = snn_layer_spike_rates(snn_model, encoder, csi, rate_idx)
    sizes = layer_sizes(snn_model)
    print(f"layer sizes (in, out): {sizes}")
    print(f"per-layer input spike rates: {[f'{r:.4f}' for r in spike_rates]}")

    total_snn_pj, total_ann_pj = 0.0, 0.0
    for (in_f, out_f), rate in zip(sizes, spike_rates):
        ops_per_timestep = in_f * out_f
        snn_pj = ops_per_timestep * t_win * E_AC_PJ * rate
        ann_pj = ops_per_timestep * t_win * E_MAC_PJ  # always dense, rate=1.0
        total_snn_pj += snn_pj
        total_ann_pj += ann_pj

    # per-example energy (the loop above already summed across T; per-batch-example, not per-batch)
    print()
    print(f"Estimated energy per window (T={t_win} timesteps), per example, synaptic layers only:")
    print(f"  SNN (spike-gated AC, {E_AC_PJ} pJ/op):  {total_snn_pj/1e3:.2f} nJ")
    print(f"  ANN (dense MAC, {E_MAC_PJ} pJ/op):       {total_ann_pj/1e3:.2f} nJ")
    print(f"  ANN / SNN ratio: {total_ann_pj/total_snn_pj:.1f}x more energy for the dense ANN")


if __name__ == "__main__":
    main()
