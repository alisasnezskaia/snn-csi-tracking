"""Non-spiking ANN twin of models.regression_snn.CSIConvSpikingRegressor,
built for the paper's energy-estimate comparison (SNN vs conventional DNN).

Deliberately NOT a fancier architecture (e.g. GRU/LSTM) -- that would
confound the comparison with an unrelated capacity/gating difference. This
uses the exact same PerFrameConvEncoder frontend, the exact same Linear
layer sizes, and the exact same leaky-accumulation recurrence (mem = beta*
mem + W @ input) as the SNN, changing only the hidden nonlinearity: ReLU on
the continuous membrane value, instead of a spike threshold + surrogate
gradient. Same weights-shape, same recurrence, dense activations instead of
sparse binary ones -- isolates exactly the "spiking vs not" difference the
energy estimate needs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_csi_tracking.models.regression_snn import PerFrameConvEncoder


class CSIConvANNRegressor(nn.Module):
    def __init__(
        self,
        num_antennas: int,
        num_subcarriers: int,
        conv_channels: list[int],
        hidden_sizes: list[int],
        output_size: int = 1,
        beta: float = 0.9,
        kernel_size: int = 9,
        num_rates: int | None = None,
        rate_embed_dim: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.conv_encoder = PerFrameConvEncoder(num_antennas, num_subcarriers, conv_channels, kernel_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.rate_embedding = nn.Embedding(num_rates, rate_embed_dim) if num_rates else None
        extra_dim = rate_embed_dim if num_rates else 0

        sizes = [self.conv_encoder.out_features + extra_dim, *hidden_sizes]
        self.hidden_layers = nn.ModuleList()
        self.hidden_sizes = hidden_sizes
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            self.hidden_layers.append(nn.Linear(in_size, out_size))

        self.readout_layer = nn.Linear(sizes[-1], output_size)

    def forward(
        self, csi: torch.Tensor, spike_encoder: nn.Module | None = None, rate_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """csi: (batch, T, NumAntennas, NumSubcarriers) normalized amplitude.
        `spike_encoder` accepted (and ignored) only so this drops into
        train_regression.py's train/evaluate loop unchanged -- the ANN has
        no spike encoding step, it consumes the conv encoder's continuous
        features directly.
        Returns: (batch, output_size) continuous prediction.
        """
        features = self.conv_encoder(csi)  # (batch, T, feat), continuous
        batch_size, num_steps, _ = features.shape
        device = csi.device
        hidden_mem = [torch.zeros(batch_size, size, device=device) for size in self.hidden_sizes]
        readout_mem = torch.zeros(batch_size, self.readout_layer.out_features, device=device)

        rate_vec = self.rate_embedding(rate_idx) if self.rate_embedding is not None else None

        for t in range(num_steps):
            cur = features[:, t, :]
            if rate_vec is not None:
                cur = torch.cat([cur, rate_vec], dim=-1)
            for i, layer in enumerate(self.hidden_layers):
                hidden_mem[i] = self.beta * hidden_mem[i] + layer(cur)
                cur = F.relu(hidden_mem[i])
                if self.dropout is not None:
                    cur = self.dropout(cur)
            readout_mem = self.beta * readout_mem + self.readout_layer(cur)

        return readout_mem
