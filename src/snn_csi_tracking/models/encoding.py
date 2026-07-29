"""Encode continuous, per-timestep CSI features into spike trains.

Input convention throughout: (batch, T, F) float tensor, already normalized
(see data.preprocessing.normalize), where F = NumSubcarriers * NumAntennas.
Output: (batch, T, F) binary spike tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from snntorch import spikegen


class RateEncoder(nn.Module):
    """Poisson rate coding: spike probability proportional to feature magnitude."""

    def __init__(self, gain: float = 1.0):
        super().__init__()
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # spikegen.rate expects values roughly in [0, 1] as spike probabilities.
        normalized = torch.sigmoid(self.gain * x)
        return spikegen.rate(normalized, num_steps=1).squeeze(0) if normalized.dim() == 3 else (
            torch.rand_like(normalized) < normalized
        ).float()


class DeltaEncoder(nn.Module):
    """Threshold-crossing (delta modulation) coding: spike when |Δfeature| > threshold.

    Better suited than rate coding for CSI, where the informative signal is in
    how the channel changes over time rather than its absolute magnitude.
    """

    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return spikegen.delta(x, threshold=self.threshold, off_spike=False)
