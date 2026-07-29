"""LSTM twin of models.presence_position_snn.SNNPresencePositionConvPerFrame
-- a fairer conventional-DL benchmark than models.presence_position_ann.
ANNPresencePositionConvPerFrame's hand-rolled leaky-accumulation recurrence
(chosen there specifically to mirror the SNN's structure for a clean energy
comparison, at the cost of stability -- see conversation: its RMSE curve
oscillated past 2x its starting error over training, no reset mechanism to
bound the membrane).

Same PerFrameConvEncoder frontend, same two-layer depth, same two heads
(presence + position) -- but the temporal recurrence is a real 2-layer
nn.LSTM (proper forget/input/output gating, designed specifically to avoid
the instability the simpler ANN twin showed) instead of either LIF spiking
or leaky-accumulation-plus-ReLU. Not part of the energy comparison (LSTMs
don't have a standard spike-vs-dense-op energy story the way the ANN twin
does) -- this is purely an accuracy/stability benchmark, addressing the
concern that comparing the SNN only against a non-standard, admittedly
unstable ANN understates how a properly-regularized conventional recurrent
model would do.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from snn_csi_tracking.models.regression_snn import PerFrameConvEncoder


class LSTMPresencePositionConvPerFrame(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_subcarriers: int,
        conv_channels: list[int],
        h1: int = 128,
        h2: int = 32,
        out_dim: int = 2,
        kernel_size: int = 9,
    ):
        super().__init__()
        self.conv_encoder = PerFrameConvEncoder(num_channels, num_subcarriers, conv_channels, kernel_size)
        in_features = self.conv_encoder.out_features
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=h1, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=h1, hidden_size=h2, num_layers=1, batch_first=True)
        self.fc_presence = nn.Linear(h2, 1)
        self.fc_position = nn.Sequential(nn.Linear(h2, 16), nn.ReLU(), nn.Linear(16, out_dim))

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """windows: (batch, NumChannels, NumSubcarriers, T), same convention
        as SNNPresencePositionConvPerFrame.forward.
        Returns (presence_seq (T, batch), position_seq (T, batch, out_dim)),
        same convention as the SNN/ANN twins."""
        conv_in = windows.permute(0, 3, 1, 2)  # (batch, T, chan, sub)
        features = self.conv_encoder(conv_in)  # (batch, T, feat), continuous
        h1_seq, _ = self.lstm(features)  # (batch, T, h1)
        h2_seq, _ = self.lstm2(h1_seq)  # (batch, T, h2)
        presence_seq = self.fc_presence(h2_seq).squeeze(-1)  # (batch, T)
        position_seq = self.fc_position(h2_seq)  # (batch, T, out_dim)
        return presence_seq.permute(1, 0), position_seq.permute(1, 0, 2)
