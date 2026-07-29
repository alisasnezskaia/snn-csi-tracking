"""Two-head LIF network: shared trunk -> presence (binary) + position (2D).

Architecture matches the notebook's spec (2 LIF layers, 128/32 units).
Class-weighted BCE loss on the presence head is needed at training time to
stop the model collapsing to "always predict present" -- see
train_presence_position.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from snn_csi_tracking.models.regression_snn import PerFrameConvEncoder, TimeAwareConvEncoder


def to_spikes(windows: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """Delta/threshold spike encoding: 1 where amplitude change >= threshold.

    windows: (batch, NumAntennas, NumSubcarriers, T), already z-scored
    amplitude (see presence_position_dataset.build_dataset).
    Returns: (T, batch, NumAntennas*NumSubcarriers) spike tensor, ready to
    feed to SNNPresencePosition.forward's per-timestep loop.
    """
    batch = windows.shape[0]
    t = windows.shape[-1]
    flat = windows.reshape(batch, -1, t)
    diff = torch.diff(flat, dim=-1, prepend=flat[..., :1])
    spikes = (diff >= threshold).float()
    return spikes.permute(2, 0, 1)


def to_spikes_and_delta(
    windows: torch.Tensor, threshold: float = 0.3, delta_clip: float = 3.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Like to_spikes, but also returns the raw signed frame-to-frame delta
    itself (clipped to +-delta_clip, scaled to roughly unit range) as a
    companion continuous tensor -- the binary spike alone only encodes
    *whether* a subcarrier's amplitude changed by more than `threshold`,
    discarding *how much* and in *which direction*, which is where motion-
    direction/speed (Doppler-like) information would live (see
    conversation). Meant to be concatenated with the spike tensor along the
    feature axis before the first linear layer (doubling in_features), not
    fed to a spiking neuron on its own -- it's a continuous companion
    signal, not itself a spike train.

    Returns (spikes, delta), both (T, batch, NumChannels*NumSubcarriers).
    """
    batch = windows.shape[0]
    t = windows.shape[-1]
    flat = windows.reshape(batch, -1, t)
    diff = torch.diff(flat, dim=-1, prepend=flat[..., :1])
    spikes = (diff >= threshold).float()
    delta = diff.clamp(-delta_clip, delta_clip) / delta_clip
    return spikes.permute(2, 0, 1), delta.permute(2, 0, 1)


class SNNPresencePosition(nn.Module):
    def __init__(self, in_features: int, h1: int = 128, h2: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.lif1 = snn.Leaky(beta=0.9, threshold=1.0)
        self.fc2 = nn.Linear(h1, h2)
        self.lif2 = snn.Leaky(beta=0.9, threshold=1.0)
        self.fc_presence = nn.Linear(h2, 1)
        self.fc_position = nn.Sequential(nn.Linear(h2, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x_seq: (T, batch, in_features) spike tensor (see to_spikes)."""
        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        presence_logit, position = None, None
        for t in range(x_seq.shape[0]):
            spk1, mem1 = self.lif1(self.fc1(x_seq[t]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            presence_logit = self.fc_presence(spk2)
            position = self.fc_position(spk2)
        return presence_logit.squeeze(-1), position


class SNNPresencePositionPerFrame(nn.Module):
    """Same trunk as SNNPresencePosition, but reads out presence/position at
    *every* internal timestep instead of only the last -- dense, frame-level
    supervision (50ms resolution) instead of one averaged label per window.
    `out_dim` is 2 for (x, y) or 3 to include depth (see depth_extraction.py)."""

    def __init__(self, in_features: int, h1: int = 128, h2: int = 32, out_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.lif1 = snn.Leaky(beta=0.9, threshold=1.0)
        self.fc2 = nn.Linear(h1, h2)
        self.lif2 = snn.Leaky(beta=0.9, threshold=1.0)
        self.fc_presence = nn.Linear(h2, 1)
        self.fc_position = nn.Sequential(nn.Linear(h2, 16), nn.ReLU(), nn.Linear(16, out_dim))

    def forward(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x_seq: (T, batch, in_features) spike tensor (see to_spikes).
        Returns (presence_seq (T, batch), position_seq (T, batch, out_dim))."""
        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        presence_seq, position_seq = [], []
        for t in range(x_seq.shape[0]):
            spk1, mem1 = self.lif1(self.fc1(x_seq[t]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            presence_seq.append(self.fc_presence(spk2))
            position_seq.append(self.fc_position(spk2))
        return torch.stack(presence_seq).squeeze(-1), torch.stack(position_seq)


class _GradientReversal(torch.autograd.Function):
    """Identity in the forward pass, negates (and scales by lambd) the
    gradient in the backward pass -- Ganin & Lempitsky, 2015. Lets a domain
    classifier attached downstream be trained normally (minimize its own
    loss) while the upstream trunk feeding it gets pushed in the OPPOSITE
    direction: towards making that classifier's job harder. Two competing
    objectives implemented as one forward-transparent op, no separate
    min-max training loop needed."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, lambd)


class SNNPresencePositionConvPerFrame(nn.Module):
    """Like SNNPresencePositionPerFrame, but with a PerFrameConvEncoder
    frontend (regression_snn.py, already validated by the grid-motion
    pipeline) instead of a flat Linear layer directly on the raw
    (channels x subcarriers) input.

    Why: a conv shares weights across subcarrier positions instead of
    giving every subcarrier x channel x hidden-unit combination its own
    independent weight -- this pipeline's flat fc1 massively overparameterizes
    the first layer relative to available data, which is exactly what made
    the CIR/delta-rate feature additions overfit so badly (see conversation).
    A conv also encodes the physical prior that neighboring subcarriers have
    correlated channel response, which a flat Linear layer has no choice but
    to re-learn from scratch, needing far more data to do it reliably.

    Unlike SNNPresencePositionPerFrame (fed pre-computed spikes via
    to_spikes), this takes raw per-frame features directly and does the
    delta/spike encoding itself, *after* the conv -- the conv's compact
    per-frame output is what gets delta-encoded into spikes, not the raw
    amplitude (same order CSIConvSpikingRegressor uses, for the same
    reason: encoding directly off 1024 raw subcarriers first would give the
    conv nothing but binary noise to work with).

    num_domains (optional): adds a domain-adversarial branch (Ganin &
    Lempitsky, 2015) off the same shared spk2 representation the presence/
    position heads use -- a small classifier predicting WHICH training
    session produced a window, wired through grad_reverse so the shared
    trunk is penalized for making that easy. Built directly in response to
    a diagnosed failure mode (see conversation): leave-activity-out cross-
    validation measured presence AUROC~0.508 (chance) with very low
    variance across every fold, consistent with the trunk encoding session
    identity rather than genuine occupancy. When num_domains is None
    (default), this is inert -- forward() behaves exactly as before, so
    existing checkpoints/callers are unaffected.

    predict_los (optional): adds a THIRD head predicting LoS/NLoS (shield
    present or not) off the same shared spk2 representation, and gates the
    position output by the presence head's own probability -- final
    position = sigmoid(presence_logit) * raw_position, so a window the
    model itself believes is empty reports a near-zero position rather than
    a spurious value. Built in direct response to a diagnosed confound (see
    conversation): the >4x noise-floor gap measured between NLoS_E and
    PLoS_E (same activity, same ground truth) is a STRUCTURED, already-
    labeled difference (shield blocks the direct path or doesn't), not
    unexplained per-session drift -- every leave-activity-out CV fold mixed
    both conditions across train/val/test without ever telling the model
    which propagation regime applied, asking it to learn one unified
    mapping across two physically different regimes at once. Making LoS/
    NLoS an explicit auxiliary output (rather than either ignoring it or
    training two fully separate per-condition models) keeps a single shared
    backbone while still letting the network structure its representation
    around the condition. Default False is inert -- forward() behaves
    exactly as before.

    Measured empirically NOT to work well (see conversation): the LoS/NLoS
    auxiliary head only reached 0.588+/-0.104 accuracy on a binary task
    (0.5 = chance) -- trying to recover a STATIC channel property from deep
    inside a pipeline built entirely around DYNAMIC, delta-encoded, motion-
    based features fights the architecture. A trivial logistic regression
    on the time-averaged raw amplitude spectrum (no motion features at all)
    got 100% accuracy on the identical leave-activity-out folds. Use
    `condition_embed_dim` below instead -- feed that (essentially free,
    always-correct) condition in as a KNOWN INPUT rather than asking this
    model to infer it as an output.

    condition_embed_dim (optional): learned nn.Embedding(2, dim) for the
    LoS/NLoS condition, concatenated to the delta-encoded spike vector at
    every timestep -- same conditioning pattern as regression_snn.py's
    rate_embedding, just for condition instead of capture rate. Requires
    `condition_idx` at forward() time. None (default) is inert.

    num_static_channels (optional): the LAST `num_static_channels` channels
    of `windows` are treated as a STATIC presence signal (e.g.
    presence_position_dataset.empty_baseline_deviation_feature) instead of
    a dynamic/motion one, and bypass delta-encoding entirely -- see
    conversation: EVERY feature in this model, without this, gets
    `torch.diff`'d against the previous timestep before the LIF layers ever
    see it (see forward()) -- exactly right for motion features, but it
    means a constant-over-time "a body is physically here right now" signal
    (from a person sitting still) diffs to ~zero and is structurally
    invisible to the network, no matter how good the raw feature is. This
    routes those channels through their own small conv encoder (no time
    mixing needed -- see `static_conv_channels`) and concatenates the RAW
    per-timestep output straight into the LIF input alongside the spikes,
    the same "concatenate a raw, non-spike-encoded vector every timestep"
    pattern already used above for condition_embed_dim. Default 0 is inert
    -- forward() behaves exactly as before.
    """

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
        threshold: float = 1.0,
        delta_threshold: float = 0.3,
        num_domains: int | None = None,
        predict_los: bool = False,
        condition_embed_dim: int | None = None,
        encoder_type: str = "perframe",
        num_static_channels: int = 0,
        static_conv_channels: list[int] | None = None,
        static_embed_dim: int = 8,
    ):
        super().__init__()
        num_dynamic_channels = num_channels - num_static_channels
        if encoder_type == "perframe":
            self.conv_encoder = PerFrameConvEncoder(num_dynamic_channels, num_subcarriers, conv_channels, kernel_size)
        elif encoder_type == "timeaware":
            # real 2D conv across subcarrier AND time jointly -- see
            # TimeAwareConvEncoder's docstring; this was the single best
            # lever found in the whole investigation as a standalone
            # presence classifier (AUROC=0.650+/-0.135), tested here
            # integrated into the actual dual-head architecture instead of
            # in isolation.
            self.conv_encoder = TimeAwareConvEncoder(num_dynamic_channels, num_subcarriers, conv_channels, sub_kernel=kernel_size)
        else:
            raise ValueError(f"unknown encoder_type: {encoder_type!r}")
        self.delta_threshold = delta_threshold
        # Straight-through surrogate gradient (snntorch's own fast_sigmoid,
        # already used elsewhere in this codebase for LIF spiking -- see
        # regression_snn.CSIConvSpikingRegressor) -- see conversation: the
        # plain `(diff >= threshold).float()` comparison this replaces has
        # EXACTLY ZERO gradient (confirmed empirically: conv_encoder's
        # weight.grad was None after a full forward+backward pass), so
        # conv_encoder never trained at all, stuck at random init for every
        # SNN experiment run before this fix. Forward pass is unchanged
        # (still a hard threshold, still binary spikes); only the backward
        # pass differs, letting gradient reach the conv for the first time.
        self.spike_grad = surrogate.fast_sigmoid()
        self.condition_embedding = nn.Embedding(2, condition_embed_dim) if condition_embed_dim else None

        self.num_static_channels = num_static_channels
        if num_static_channels > 0:
            self.static_encoder = PerFrameConvEncoder(
                num_static_channels, num_subcarriers, static_conv_channels or [8, 16], kernel_size
            )
            self.static_proj = nn.Linear(self.static_encoder.out_features, static_embed_dim)
        else:
            self.static_encoder = None
            self.static_proj = None

        in_features = (
            self.conv_encoder.out_features + (condition_embed_dim or 0)
            + (static_embed_dim if num_static_channels > 0 else 0)
        )
        self.fc1 = nn.Linear(in_features, h1)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc2 = nn.Linear(h1, h2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc_presence = nn.Linear(h2, 1)
        self.fc_position = nn.Sequential(nn.Linear(h2, 16), nn.ReLU(), nn.Linear(16, out_dim))
        self.num_domains = num_domains
        if num_domains:
            self.fc_domain = nn.Sequential(nn.Linear(h2, 16), nn.ReLU(), nn.Linear(16, num_domains))
        self.predict_los = predict_los
        if predict_los:
            self.fc_los = nn.Linear(h2, 1)

    def forward(self, windows: torch.Tensor, grl_lambda: float = 1.0, condition_idx: torch.Tensor | None = None):
        """windows: (batch, NumChannels, NumSubcarriers, T), already
        z-scored amplitude+phase[+extra] features (see
        presence_position_dataset.build_perframe_dataset) -- raw features,
        NOT pre-spike-encoded (unlike SNNPresencePositionPerFrame).
        condition_idx: (batch,) long tensor, 0/1 for NLoS/PLoS -- required
        iff this model was built with condition_embed_dim set (see class
        docstring); ignored otherwise.
        Returns (presence_seq (T, batch), position_seq (T, batch, out_dim)),
        same convention as SNNPresencePositionPerFrame -- plus, in this
        order when enabled: los_seq (T, batch) if predict_los, then
        domain_seq (T, batch, num_domains) if num_domains. position_seq is
        already gated by presence probability when predict_los is set (see
        class docstring) -- callers don't need to re-mask it themselves."""
        t = windows.shape[-1]
        if self.num_static_channels > 0:
            dynamic_windows = windows[:, : -self.num_static_channels]
            static_windows = windows[:, -self.num_static_channels :]
        else:
            dynamic_windows, static_windows = windows, None

        conv_in = dynamic_windows.permute(0, 3, 1, 2)  # (batch, T, chan, sub)
        features = self.conv_encoder(conv_in)  # (batch, T, feat)
        diff = torch.diff(features, dim=1, prepend=features[:, :1])
        spikes = self.spike_grad(diff - self.delta_threshold)  # (batch, T, feat) -- forward: hard threshold,
                                                                 # backward: fast-sigmoid surrogate (see __init__)

        static_embed = None
        if static_windows is not None:
            static_conv_in = static_windows.permute(0, 3, 1, 2)  # (batch, T, static_chan, sub)
            static_feat = self.static_encoder(static_conv_in)  # (batch, T, static_out_features) -- NOT delta-encoded
            static_embed = self.static_proj(static_feat)  # (batch, T, static_embed_dim)

        condition_vec = self.condition_embedding(condition_idx) if self.condition_embedding is not None else None

        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        presence_seq, position_seq, los_seq, domain_seq = [], [], [], []
        for step in range(t):
            cur = spikes[:, step]
            if static_embed is not None:
                cur = torch.cat([cur, static_embed[:, step]], dim=-1)
            if condition_vec is not None:
                cur = torch.cat([cur, condition_vec], dim=-1)
            spk1, mem1 = self.lif1(self.fc1(cur), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            presence_logit = self.fc_presence(spk2)
            raw_position = self.fc_position(spk2)
            if self.predict_los:
                presence_prob = torch.sigmoid(presence_logit)
                position = presence_prob * raw_position
                los_seq.append(self.fc_los(spk2))
            else:
                position = raw_position
            presence_seq.append(presence_logit)
            position_seq.append(position)
            if self.num_domains:
                domain_seq.append(self.fc_domain(grad_reverse(spk2, grl_lambda)))

        out = [torch.stack(presence_seq).squeeze(-1), torch.stack(position_seq)]
        if self.predict_los:
            out.append(torch.stack(los_seq).squeeze(-1))
        if self.num_domains:
            out.append(torch.stack(domain_seq))
        return tuple(out)
