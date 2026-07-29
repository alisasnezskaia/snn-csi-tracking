"""Convolutional SNN (CSNN) presence classifier -- a genuinely spiking
version of image_cnn_presence.ImageCNNPresenceClassifier, keeping the same
advantage that made that plain-CNN test the best result of the night
(AUROC=0.650+/-0.135): real 2D convolution spanning BOTH the subcarrier
axis and SEVERAL timesteps at once, unlike PerFrameConvEncoder (used by
the SNN/ANN twins), which only ever convolves one frame at a time and can
only combine information across time later, slowly, through LIF/leaky-
accumulation recurrence.

Key design choice: the 2D conv pools the (huge, 1024-wide, redundant)
subcarrier axis aggressively, but leaves the TIME axis (t_win) untouched --
so the conv's own output is still a genuine per-timestep SEQUENCE, not one
collapsed static vector. That sequence is then delta/spike-encoded (same
convention as SNNPresencePositionConvPerFrame: encode the CONV's compact
features, not raw amplitude, into spikes) and fed through LIF neurons --
so this keeps the same spike-gated, sparse-op energy story the paper's
efficiency claim depends on, while ALSO letting the conv kernel see
several original timesteps jointly per application (kernel_size time-span
> 1), the thing a per-frame-only conv structurally cannot do.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn


class ConvSNNPresenceClassifier(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_subcarriers: int,
        t_win: int,
        conv_channels: tuple[int, ...] = (16, 32, 64),
        time_kernel: int = 5,
        sub_kernel: int = 9,
        sub_pool: int = 4,
        h1: int = 128,
        beta: float = 0.9,
        threshold: float = 1.0,
        delta_threshold: float = 0.3,
    ):
        super().__init__()
        layers = []
        in_ch = num_channels
        for out_ch in conv_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=(sub_kernel, time_kernel),
                                     padding=(sub_kernel // 2, time_kernel // 2)))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d((sub_pool, 1)))  # pool subcarrier axis ONLY -- keep full time resolution
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, num_subcarriers, t_win)
            conv_out = self.conv(dummy)  # (1, C, reduced_sub, t_win) -- time axis untouched
            self.feat_per_step = conv_out.shape[1] * conv_out.shape[2]

        self.delta_threshold = delta_threshold
        self.fc1 = nn.Linear(self.feat_per_step, h1)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc_presence = nn.Linear(h1, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """windows: (batch, NumChannels, NumSubcarriers, T_win).
        Returns (batch,) -- ONE presence logit per whole window (mean over
        the internal per-timestep spiking sequence), matching
        ImageCNNPresenceClassifier's convention for a direct comparison."""
        conv_out = self.conv(windows)  # (batch, C, reduced_sub, t_win)
        b, c, s, t = conv_out.shape
        seq = conv_out.permute(0, 3, 1, 2).reshape(b, t, c * s)  # (batch, t_win, feat_per_step)

        diff = torch.diff(seq, dim=1, prepend=seq[:, :1])
        spikes = (diff >= self.delta_threshold).float()

        mem1 = self.lif1.init_leaky()
        presence_seq = []
        for step in range(t):
            spk1, mem1 = self.lif1(self.fc1(spikes[:, step]), mem1)
            presence_seq.append(self.fc_presence(spk1))
        return torch.stack(presence_seq, dim=1).squeeze(-1).mean(dim=1)
