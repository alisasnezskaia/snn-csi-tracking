"""Non-spiking ANN twin of models.presence_position_snn.
SNNPresencePositionConvPerFrame, built for the paper's energy-estimate
comparison (SNN vs conventional DNN) on the actual joint presence+position
task (not the scalar regression task -- see models/regression_ann.py for
that one).

Same rationale as regression_ann.py: identical PerFrameConvEncoder
frontend, identical Linear layer sizes, identical leaky-accumulation
recurrence (mem = beta*mem + W @ input) as the SNN, changing only the
hidden nonlinearity (ReLU on the continuous membrane value instead of a
spike threshold + surrogate gradient) and skipping delta/spike encoding
entirely -- the ANN consumes the conv encoder's continuous per-frame
features directly. Same two heads (presence logit + position), same
per-timestep readout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_csi_tracking.models.regression_snn import PerFrameConvEncoder


class ANNPresencePositionConvPerFrame(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_subcarriers: int,
        conv_channels: list[int],
        h1: int = 128,
        h2: int = 32,
        out_dim: int = 2,
        kernel_size: int = 9,
        beta: float = 0.9,
    ):
        super().__init__()
        self.beta = beta
        self.conv_encoder = PerFrameConvEncoder(num_channels, num_subcarriers, conv_channels, kernel_size)
        in_features = self.conv_encoder.out_features
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc_presence = nn.Linear(h2, 1)
        self.fc_position = nn.Sequential(nn.Linear(h2, 16), nn.ReLU(), nn.Linear(16, out_dim))

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """windows: (batch, NumChannels, NumSubcarriers, T), same convention
        as SNNPresencePositionConvPerFrame.forward. No delta/spike encoding
        -- the conv encoder's continuous per-frame output feeds fc1 directly.
        Returns (presence_seq (T, batch), position_seq (T, batch, out_dim)),
        same convention as the SNN twin."""
        t = windows.shape[-1]
        conv_in = windows.permute(0, 3, 1, 2)  # (batch, T, chan, sub)
        features = self.conv_encoder(conv_in)  # (batch, T, feat), continuous
        batch_size = features.shape[0]
        device = features.device

        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=device)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=device)
        presence_seq, position_seq = [], []
        for step in range(t):
            cur = features[:, step]
            mem1 = self.beta * mem1 + self.fc1(cur)
            spk1 = F.relu(mem1)
            mem2 = self.beta * mem2 + self.fc2(spk1)
            spk2 = F.relu(mem2)
            presence_seq.append(self.fc_presence(spk2))
            position_seq.append(self.fc_position(spk2))
        return torch.stack(presence_seq).squeeze(-1), torch.stack(position_seq)
