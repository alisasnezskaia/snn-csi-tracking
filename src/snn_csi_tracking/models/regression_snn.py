"""Conv frontend (per-frame frequency features) + spiking backend, regression head.

Architecture for the motion/displacement regression task: absolute (x,y)
position isn't reliably recoverable from a single 3-antenna link, so the
target is "how much/fast is the person moving" instead.

Two deliberate design choices, carried over from the presence/position
classification architecture:

1. The conv only looks at ONE frame at a time (subcarrier axis only, time axis
   untouched) -- unlike Wi-Spike's joint spatio-temporal spiking conv, which
   is well-suited to a single per-window classification label but would
   collapse away the fine-grained temporal detail a continuous regression
   target over time actually needs.
2. The final readout uses the last LIF layer's membrane potential, not a
   spike-rate/count (spike rate is a natural fit for a bounded class score,
   not for an unbounded continuous regression value).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class PerFrameConvEncoder(nn.Module):
    """1D conv over the subcarrier axis, applied independently to each frame.

    Input:  (batch, T, NumAntennas, NumSubcarriers)
    Output: (batch, T, out_features) -- same T, compact per-frame features.
    """

    def __init__(
        self,
        num_antennas: int,
        num_subcarriers: int,
        conv_channels: list[int],
        kernel_size: int = 9,
    ):
        super().__init__()
        layers = []
        in_ch = num_antennas
        for out_ch in conv_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, num_antennas, num_subcarriers)
            self.out_features = self.conv(dummy).flatten(1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, t, ant, sub = x.shape
        x = x.reshape(batch * t, ant, sub)
        x = self.conv(x)
        x = x.flatten(1)
        return x.reshape(batch, t, -1)


class TimeAwareConvEncoder(nn.Module):
    """Like PerFrameConvEncoder, but a genuine 2D conv spanning BOTH the
    subcarrier axis AND several adjacent TIMESTEPS jointly, instead of
    treating every frame independently. Standalone presence-classifier
    experiments (image_cnn_presence.py, AUROC=0.650+/-0.135, best fold
    0.802) found this the strongest single lever tried, because it lets
    one convolution directly correlate a short stretch of time with a
    band of frequencies, something PerFrameConvEncoder structurally
    cannot do (it only combines time information afterward, slowly,
    through whatever recurrence sits downstream). This integrates that
    same idea as a DROP-IN replacement for PerFrameConvEncoder -- same
    (batch, T, NumAntennas, NumSubcarriers)
    input, same (batch, T, out_features) output -- so it plugs into the
    real dual-head (presence+position) architecture instead of only being
    tested in isolation.

    Pools the subcarrier axis aggressively (1024 is huge and highly
    redundant across nearby subcarriers) but leaves the TIME axis
    untouched, so downstream LIF/recurrent layers still get one genuine
    output per original input frame, not a collapsed window-level summary.
    """

    def __init__(
        self,
        num_antennas: int,
        num_subcarriers: int,
        conv_channels: list[int],
        time_kernel: int = 5,
        sub_kernel: int = 9,
        sub_pool: int = 4,
    ):
        super().__init__()
        layers = []
        in_ch = num_antennas
        for out_ch in conv_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=(sub_kernel, time_kernel),
                                     padding=(sub_kernel // 2, time_kernel // 2)))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d((sub_pool, 1)))  # subcarrier axis only -- time axis untouched
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, num_antennas, num_subcarriers, 64)  # placeholder T -- out_features doesn't depend on it
            conv_out = self.conv(dummy)
            self.out_features = conv_out.shape[1] * conv_out.shape[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, NumAntennas, NumSubcarriers) -- same convention as
        PerFrameConvEncoder.forward. Returns (batch, T, out_features)."""
        x = x.permute(0, 2, 3, 1)  # (batch, ant, sub, T) -- ant=conv channels, (sub, T) = the 2D spatial axes
        x = self.conv(x)  # (batch, C, reduced_sub, T)
        b, c, s, t = x.shape
        return x.permute(0, 3, 1, 2).reshape(b, t, c * s)  # (batch, T, out_features)


class CSIConvSpikingRegressor(nn.Module):
    """PerFrameConvEncoder -> (caller-supplied spike encoder) -> LIF stack ->
    continuous readout (final membrane potential, summed... no: taken at the
    last timestep -- see forward()).
    """

    def __init__(
        self,
        num_antennas: int,
        num_subcarriers: int,
        conv_channels: list[int],
        hidden_sizes: list[int],
        output_size: int = 1,
        beta: float = 0.9,
        threshold: float = 1.0,
        kernel_size: int = 9,
        num_rates: int | None = None,
        rate_embed_dim: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.conv_encoder = PerFrameConvEncoder(num_antennas, num_subcarriers, conv_channels, kernel_size)
        # Applied to each hidden LIF layer's spike output, at every timestep,
        # before it feeds the next layer -- NOT before the DeltaEncoder (would
        # zero random features independently per timestep, which reads to the
        # delta encoder as spurious value transitions and injects fake spikes
        # into the one signal it's supposed to capture cleanly).
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # rate_embed_dim gets concatenated onto every timestep's spike vector
        # (not delta-encoded itself -- it's constant per-window, so folding it
        # into the pre-spike features would just delta to zero and vanish).
        # This is how the CSI capture's interframe rate (5/10/50/100ms -- same
        # window length in samples, wildly different real-world duration) gets
        # surfaced to the model instead of being silently discarded.
        self.rate_embedding = nn.Embedding(num_rates, rate_embed_dim) if num_rates else None
        extra_dim = rate_embed_dim if num_rates else 0

        sizes = [self.conv_encoder.out_features + extra_dim, *hidden_sizes]
        self.hidden_layers = nn.ModuleList()
        self.hidden_lifs = nn.ModuleList()
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            self.hidden_layers.append(nn.Linear(in_size, out_size))
            self.hidden_lifs.append(snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad))

        self.readout_layer = nn.Linear(sizes[-1], output_size)
        self.readout_lif = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, reset_mechanism="none")

    def forward(
        self, csi: torch.Tensor, spike_encoder: nn.Module, rate_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """csi: (batch, T, NumAntennas, NumSubcarriers) normalized amplitude.
        spike_encoder: e.g. models.encoding.DeltaEncoder, applied to the
        conv-encoder's compact per-frame features (not the raw CSI).
        rate_idx: (batch,) long tensor indexing into RATES, required iff this
        model was built with num_rates set.
        Returns: (batch, output_size) continuous prediction.
        """
        features = self.conv_encoder(csi)  # (batch, T, feat)
        spikes = spike_encoder(features)  # (batch, T, feat) binary

        batch_size, num_steps, _ = spikes.shape
        hidden_mem = [lif.init_leaky() for lif in self.hidden_lifs]
        readout_mem = self.readout_lif.init_leaky()

        rate_vec = self.rate_embedding(rate_idx) if self.rate_embedding is not None else None

        for t in range(num_steps):
            cur = spikes[:, t, :]
            if rate_vec is not None:
                cur = torch.cat([cur, rate_vec], dim=-1)
            for i, (layer, lif) in enumerate(zip(self.hidden_layers, self.hidden_lifs)):
                cur, hidden_mem[i] = lif(layer(cur), hidden_mem[i])
                if self.dropout is not None:
                    cur = self.dropout(cur)
            _, readout_mem = self.readout_lif(self.readout_layer(cur), readout_mem)

        return readout_mem
