"""Non-spiking baseline for the motion regression task: same conv frontend as
CSIConvSpikingRegressor (regression_snn.py), but a standard GRU instead of a
delta-encoded LIF stack for the temporal part, and no spike encoding at all --
the conv features feed the GRU directly as continuous values.

Exists specifically to answer "did the spiking part actually earn its keep,"
not just "is the overall pipeline better than guessing the mean" -- a real
architecture comparison needs both numbers, not just the SNN's own result.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from snn_csi_tracking.models.regression_snn import PerFrameConvEncoder


class CSIConvGRURegressor(nn.Module):
    """PerFrameConvEncoder (identical to the spiking version) -> GRU -> linear readout."""

    def __init__(
        self,
        num_antennas: int,
        num_subcarriers: int,
        conv_channels: list[int],
        hidden_size: int,
        output_size: int = 1,
        kernel_size: int = 9,
        num_layers: int = 1,
    ):
        super().__init__()
        self.conv_encoder = PerFrameConvEncoder(num_antennas, num_subcarriers, conv_channels, kernel_size)
        self.gru = nn.GRU(
            input_size=self.conv_encoder.out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        """csi: (batch, T, NumAntennas, NumSubcarriers) normalized amplitude.
        Returns: (batch, output_size) continuous prediction.
        """
        features = self.conv_encoder(csi)  # (batch, T, feat)
        _, h_n = self.gru(features)  # h_n: (num_layers, batch, hidden_size)
        return self.readout(h_n[-1])
