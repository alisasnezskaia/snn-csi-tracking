"""Leaky-integrate-and-fire spiking network for windowed CSI classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class CSISpikingNet(nn.Module):
    """Fully-connected LIF network processing a (batch, T, input_size) spike train.

    Each hidden layer is Linear -> Leaky; the readout layer's spikes are summed
    over time to form classification logits (rate-coded output).
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        num_classes: int,
        beta: float = 0.9,
        threshold: float = 1.0,
    ):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()

        sizes = [input_size, *hidden_sizes]
        self.hidden_layers = nn.ModuleList()
        self.hidden_lifs = nn.ModuleList()
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            self.hidden_layers.append(nn.Linear(in_size, out_size))
            self.hidden_lifs.append(
                snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
            )

        self.readout_layer = nn.Linear(sizes[-1], num_classes)
        self.readout_lif = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """spikes: (batch, T, input_size) -> logits: (batch, num_classes)."""
        batch_size, num_steps, _ = spikes.shape
        hidden_mem = [lif.init_leaky() for lif in self.hidden_lifs]
        readout_mem = self.readout_lif.init_leaky()

        spike_count = torch.zeros(batch_size, self.readout_layer.out_features, device=spikes.device)

        for t in range(num_steps):
            cur = spikes[:, t, :]
            for i, (layer, lif) in enumerate(zip(self.hidden_layers, self.hidden_lifs)):
                cur, hidden_mem[i] = lif(layer(cur), hidden_mem[i])
            out_spk, readout_mem = self.readout_lif(self.readout_layer(cur), readout_mem)
            spike_count = spike_count + out_spk

        return spike_count / num_steps
