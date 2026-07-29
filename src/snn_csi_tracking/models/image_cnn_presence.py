"""A genuine 2D-image CNN presence classifier -- direct test of "treat the
window as a photo" (see conversation): our existing per-window feature
array is already shaped like a multi-channel image (channels x subcarriers
x time), but every model built so far (PerFrameConvEncoder, used by both
the SNN and ANN twins) only convolves across the SUBCARRIER axis, one
frame at a time -- it never lets a single convolution see two different
timesteps together. That's a real, structural gap: it can't directly pick
up a pattern that spans time (e.g. a moving reflection's signature
shifting over both subcarrier and time together), the same way a real photo
CNN would.

This uses actual nn.Conv2d layers spanning BOTH the subcarrier and time
axes jointly. Deliberately non-spiking and simple (plain CNN, one holistic
presence judgment per whole window, not a per-frame recurrent sequence) --
this is a fast test of whether the underlying idea has merit at all, before
investing in a spiking version.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ImageCNNPresenceClassifier(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_subcarriers: int,
        t_win: int,
        conv_channels: tuple[int, ...] = (16, 32, 64),
    ):
        super().__init__()
        layers = []
        in_ch = num_channels
        for out_ch in conv_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=(9, 5), padding=(4, 2)))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d((4, 2)))  # aggressive pooling on the huge (1024) subcarrier axis, gentler on time
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, num_subcarriers, t_win)
            self.out_features = self.conv(dummy).flatten(1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.out_features, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """windows: (batch, NumChannels, NumSubcarriers, T_win) -- the SAME
        window tensor already used elsewhere, treated here as a genuine
        multi-channel image (2D conv sees subcarrier AND time jointly).
        Returns (batch,) -- ONE presence logit per whole window, not a
        per-frame sequence."""
        feat = self.conv(windows).flatten(1)
        return self.fc(feat).squeeze(-1)
