"""Shared inference helpers for the presence/position models: load a
checkpoint by "kind", and turn its window-level predictions into one
continuous per-CSI-frame stream (used by both the trajectory-comparison
plot and the spy-cam overlay video).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from snn_csi_tracking.models.presence_position_snn import SNNPresencePosition, SNNPresencePositionPerFrame, to_spikes

REPO_ROOT = Path(__file__).parent.parent.parent
RAW_ROOT = REPO_ROOT / "data" / "raw_captures"
TRAJECTORY_CACHE_DIR = REPO_ROOT / "data" / "processed" / "presence_position_trajectories"

T_WIN = 64
STRIDE = 32
DELTA_THRESHOLD = 0.3
NUM_SUBCARRIERS = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CHECKPOINTS = {
    "windowed": REPO_ROOT / "results" / "models" / "presence_position_snn_ampphase_50ms.pt",
    "perframe2d": REPO_ROOT / "results" / "models" / "snn_perframe_model_50ms.pt",
    "perframe3d": REPO_ROOT / "results" / "models" / "presence_position_snn_perframe_ampphase_3d_50ms.pt",
}

# channel count -> use_phase, matching presence_position_dataset.compute_features
_CHANNELS_TO_FEATURE_FLAGS = {3: False, 7: True}


def load_model(kind: str, checkpoint_path: Path) -> tuple[torch.nn.Module, bool]:
    """Returns (model, use_phase), inferred from the checkpoint's own input
    width (3*1024=amp-only, 7*1024=amp+phase) rather than tracked
    separately, so callers always build matching features without risk of
    mismatching a checkpoint against the wrong feature set."""
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    in_features = state_dict["fc1.weight"].shape[1]
    num_channels = in_features // NUM_SUBCARRIERS
    use_phase = _CHANNELS_TO_FEATURE_FLAGS[num_channels]

    if kind == "windowed":
        model = SNNPresencePosition(in_features=in_features)
    else:
        out_dim = 3 if kind == "perframe3d" else 2
        model = SNNPresencePositionPerFrame(in_features=in_features, out_dim=out_dim)
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval(), use_phase


def predict_dense(model, kind: str, amp_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (presence_prob(t), position_pred(t, out_dim), valid(t)) for
    every CSI frame in the trial, sliding T_WIN/STRIDE windows over it and
    averaging overlapping predictions -- windowed checkpoints place their
    one prediction per window at that window's center frame instead."""
    n = amp_z.shape[2]
    starts = list(range(0, n - T_WIN + 1, STRIDE))
    windows = np.stack([amp_z[:, :, s : s + T_WIN] for s in starts])
    x = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
    spikes = to_spikes(x, threshold=DELTA_THRESHOLD)

    with torch.no_grad():
        pres_logits, pos_pred = model(spikes)
    probs = torch.sigmoid(pres_logits).cpu().numpy()
    pos = pos_pred.cpu().numpy()
    out_dim = pos.shape[-1]

    presence_sum = np.zeros(n)
    position_sum = np.zeros((n, out_dim))
    counts = np.zeros(n)

    if kind == "windowed":
        for wi, s in enumerate(starts):
            center = s + T_WIN // 2
            presence_sum[center] += probs[wi]
            position_sum[center] += pos[wi]
            counts[center] += 1
    else:
        for wi, s in enumerate(starts):
            presence_sum[s : s + T_WIN] += probs[:, wi]
            position_sum[s : s + T_WIN] += pos[:, wi, :]
            counts[s : s + T_WIN] += 1

    valid = counts > 0
    presence = np.full(n, np.nan)
    position = np.full((n, out_dim), np.nan)
    presence[valid] = presence_sum[valid] / counts[valid]
    position[valid] = position_sum[valid] / counts[valid, None]
    return presence, position, valid


def predict_dense_conv(model, amp_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like predict_dense, but for SNNPresencePositionConvPerFrame -- that
    model does its own conv + delta/spike encoding internally on raw
    per-frame features, so it's called directly on the windowed features
    (model(x)), not pre-encoded via to_spikes first. Always "perframe"-style
    stitching (that model has no "windowed"/one-prediction-per-window mode)."""
    n = amp_z.shape[2]
    starts = list(range(0, n - T_WIN + 1, STRIDE))
    windows = np.stack([amp_z[:, :, s : s + T_WIN] for s in starts])
    x = torch.tensor(windows, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pres_logits, pos_pred = model(x)
    probs = torch.sigmoid(pres_logits).cpu().numpy()
    pos = pos_pred.cpu().numpy()
    out_dim = pos.shape[-1]

    presence_sum = np.zeros(n)
    position_sum = np.zeros((n, out_dim))
    counts = np.zeros(n)
    for wi, s in enumerate(starts):
        presence_sum[s : s + T_WIN] += probs[:, wi]
        position_sum[s : s + T_WIN] += pos[:, wi, :]
        counts[s : s + T_WIN] += 1

    valid = counts > 0
    presence = np.full(n, np.nan)
    position = np.full((n, out_dim), np.nan)
    presence[valid] = presence_sum[valid] / counts[valid]
    position[valid] = position_sum[valid] / counts[valid, None]
    return presence, position, valid


def debounce_presence(probs: np.ndarray, threshold: float = 0.5, min_run: int = 6) -> np.ndarray:
    """Hysteresis on the raw per-frame presence probability: only flips
    state after `min_run` consecutive frames vote the other way, instead of
    thresholding each frame independently. Ported from the notebook's Part
    3 fix for the exact flicker the raw per-frame signal has (see
    conversation: the un-debounced signal bounces frame to frame even
    within a single true presence/absence stretch)."""
    binary = (probs > threshold).astype(int)
    state = binary[0]
    out = np.zeros_like(binary)
    run = 0
    for i in range(len(binary)):
        if binary[i] == state:
            run = 0
        else:
            run += 1
            if run >= min_run:
                state = binary[i]
                run = 0
        out[i] = state
    return out.astype(bool)


def smooth_position(position: np.ndarray, present: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered rolling-mean smoothing of the per-frame position stream, so
    the drawn/plotted point doesn't jump frame to frame the way the raw
    per-window prediction does. Frames `present` marks False (already-
    debounced absence, or an invalid/uncovered frame) are masked to NaN
    first, so a window straddling a real absence never blends positions
    from two separate presence stretches together -- safe as long as
    `window` stays well under debounce_presence's `min_run` (the shortest
    a real stretch can be), same assumption trajectory_extraction.
    clean_trajectory makes between its smoothing window and its gap-
    preservation threshold."""
    df = pd.DataFrame(position)
    df[~present] = np.nan
    return df.rolling(window=window, center=True, min_periods=1).mean().to_numpy()
