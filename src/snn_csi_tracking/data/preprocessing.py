"""Schema-independent CSI preprocessing: works on already-loaded numpy arrays."""

from __future__ import annotations

import numpy as np


def _soft_threshold(c: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Soft-thresholding via the direct sign/abs/maximum formula, not
    pywt.threshold -- pywt's own implementation divides by the coefficient's
    magnitude internally and produces NaN (0/0) whenever both the
    coefficient and the threshold are exactly zero, which happens for every
    genuinely constant/silent channel (confirmed: CSI guard-band/null
    subcarriers, ~84 of 1024, have zero variance and so zero detail
    coefficients at every level -- see conversation). This formula has no
    division, so it's NaN-safe unconditionally."""
    return np.sign(c) * np.maximum(np.abs(c) - threshold, 0)


def wavelet_denoise(amp: np.ndarray, time_axis: int = -1, wavelet: str = "db4", level: int | None = None) -> np.ndarray:
    """Wavelet-domain denoising along `time_axis`: decomposes each time
    series into wavelet coefficients, soft-thresholds the detail
    coefficients at the universal (VisuShrink) threshold -- noise sigma
    estimated per-series from the finest detail level's median absolute
    deviation (robust to the real, non-Gaussian signal energy also present
    in that level) -- then reconstructs. Standard technique in CSI-sensing
    literature for removing hardware/environment measurement noise while
    preserving real, slower channel variation; unlike a fixed-cutoff
    low-pass filter, the threshold adapts per series.

    Args:
        amp: real-valued array, any shape, with the time series along
            `time_axis`.
        wavelet: PyWavelets wavelet name.
        level: decomposition level (None = pywt's max for this length).
    Returns:
        Denoised array, same shape as `amp`, float32.
    """
    import pywt

    n = amp.shape[time_axis]
    coeffs = pywt.wavedec(amp, wavelet, axis=time_axis, level=level)
    detail = coeffs[-1]
    mad = np.median(np.abs(detail), axis=time_axis, keepdims=True)
    sigma = mad / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(n))
    coeffs[1:] = [_soft_threshold(c, threshold) for c in coeffs[1:]]
    denoised = pywt.waverec(coeffs, wavelet, axis=time_axis)

    slicer = [slice(None)] * amp.ndim
    slicer[time_axis] = slice(0, n)
    return denoised[tuple(slicer)].astype(np.float32)


def pca_denoise(amp: np.ndarray, num_components: int, freq_axis: int, time_axis: int) -> np.ndarray:
    """PCA-based denoising across `freq_axis` (subcarriers): treats each
    timestep as one sample in subcarrier-space and keeps only the top
    `num_components` principal components (by variance) before
    reconstructing -- assumes real channel variation from motion is
    captured by a few dominant components while measurement noise spreads
    thinly across many. Common technique in CSI-based motion/activity
    sensing (e.g. CARM-style pipelines). PCA is fit independently per
    leading (e.g. antenna) index.

    Args:
        amp: real-valued array with a subcarrier axis (`freq_axis`) and a
            time axis (`time_axis`); any remaining axes (e.g. antenna) are
            treated as independent leading dimensions.
        num_components: how many principal components to keep.
    Returns:
        Denoised array, same shape as `amp`, float32.
    """
    moved = np.moveaxis(amp, [freq_axis, time_axis], [-2, -1])
    orig_shape = moved.shape
    leading = int(np.prod(orig_shape[:-2])) if moved.ndim > 2 else 1
    flat = moved.reshape(leading, orig_shape[-2], orig_shape[-1])

    out = np.empty_like(flat, dtype=np.float32)
    for i in range(leading):
        x = flat[i].T  # (T, NumSubcarriers) -- samples x features
        mean = x.mean(axis=0, keepdims=True)
        centered = x - mean
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        k = min(num_components, s.shape[0])
        recon = (u[:, :k] * s[:k]) @ vt[:k] + mean
        out[i] = recon.T

    out = out.reshape(orig_shape)
    return np.moveaxis(out, [-2, -1], [freq_axis, time_axis])


PCA_DENOISE_COMPONENTS = 10  # of 1024 subcarriers -- keep the few dominant,
                             # motion-correlated components, treat the rest
                             # as noise (see pca_denoise)


def denoise_amplitude(amp: np.ndarray, denoise: str | None, freq_axis: int, time_axis: int) -> np.ndarray:
    """Dispatches to wavelet_denoise / pca_denoise (or a no-op) by name --
    shared by presence_position_dataset.compute_features and
    raw_capture_loader.load_empty_room_baseline, so the empty-room
    reference and the live signal it's subtracted from are always denoised
    the same way (denoising one but not the other would reintroduce
    exactly the mismatch the baseline is meant to remove)."""
    if denoise is None:
        return amp
    if denoise == "wavelet":
        return wavelet_denoise(amp, time_axis=time_axis)
    if denoise == "pca":
        return pca_denoise(amp, PCA_DENOISE_COMPONENTS, freq_axis=freq_axis, time_axis=time_axis)
    raise ValueError(f"unknown denoise method: {denoise!r}")


def normalize(csi: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Amplitude-extract (if complex) and z-score normalize per subcarrier/antenna.

    Args:
        csi: array of shape (T, NumSubcarriers, NumAntennas), real or complex.
    Returns:
        float32 array of the same shape, zero mean / unit variance per (subcarrier, antenna).
    """
    amplitude = np.abs(csi) if np.iscomplexobj(csi) else csi
    amplitude = amplitude.astype(np.float32)
    mean = amplitude.mean(axis=0, keepdims=True)
    std = amplitude.std(axis=0, keepdims=True)
    return (amplitude - mean) / (std + eps)


def energy_normalize(csi: np.ndarray, freq_axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    """Amplitude-extract (if complex) and normalize each FRAME's own energy
    to 1 across `freq_axis` (subcarriers) -- unlike normalize()'s per-trial
    z-score (mean/std computed across the whole time axis, one shared
    reference for every frame) or a cross-day empty-room baseline (a
    reference recorded on a DIFFERENT day), this normalizes at the same
    granularity AGC actually operates at: frame-by-frame, not trial-by-trial
    or day-by-day.

    Motivation confirmed empirically, not just in theory: a model trained
    on per-trial z-scored amplitude produced nearly IDENTICAL raw presence
    probabilities (mean/std/max all matching to 3 decimals) on two held-out
    test sessions with opposite ground truth -- one genuinely empty, one
    65% occupied. Per-trial normalization scales every trial to mean-0/
    std-1 BY CONSTRUCTION, including genuinely silent ones, so it structurally
    cannot carry the absolute "how much is this really changing" signal
    presence detection needs on a session the model wasn't trained on. Per-
    frame energy normalization keeps the AGC gain state from leaking into
    the features (same problem the empty-room baseline was trying to solve)
    without needing a separately-recorded, potentially-drifted reference.

    Args:
        csi: array of shape (T, NumSubcarriers, NumAntennas), real or complex.
        freq_axis: which axis is the subcarrier axis (0 for (T, Sub, Ant)).
    Returns:
        float32 array of the same shape, each frame's L2 energy == 1.
    """
    amplitude = np.abs(csi) if np.iscomplexobj(csi) else csi
    amplitude = amplitude.astype(np.float32)
    energy = np.sqrt((amplitude ** 2).sum(axis=freq_axis, keepdims=True))
    return amplitude / (energy + eps)


def phase_diff_features(csi: np.ndarray, reference_antenna: int = 0, eps: float = 1e-8) -> np.ndarray:
    """Cross-antenna phase difference, as (sin, cos) pairs per non-reference antenna.

    All antennas on one receiver share the same local oscillator, so carrier/
    sampling frequency offset (CFO/SFO) drift rotates every antenna's raw
    phase by (almost) the same amount at any given instant. Comparing antenna
    `a` against `reference_antenna` cancels that shared drift without having
    to estimate and remove it explicitly -- the only thing left in the
    difference is the antennas' differing path geometry (motion, angle).

    Computed via the conjugate-product trick (csi_a * conj(csi_ref)), not a
    naive angle subtraction -- this handles the +-pi wraparound correctly by
    construction. The result is emitted as (sin, cos) of that difference
    rather than the raw angle for the same wraparound reason one level up:
    a raw angle jumps discontinuously from +pi to -pi even for an
    infinitesimal true change, which the downstream DeltaEncoder (threshold
    on frame-to-frame change) would read as a spurious near-maximal spike.
    sin/cos is the continuous, wraparound-free representation of the same
    angle.

    Args:
        csi: complex array of shape (T, NumSubcarriers, NumAntennas).
        reference_antenna: antenna index every other antenna is compared against.
    Returns:
        float32 array of shape (T, NumSubcarriers, (NumAntennas-1)*2).
    """
    ref = csi[:, :, reference_antenna]
    other_idx = [a for a in range(csi.shape[2]) if a != reference_antenna]
    channels = []
    for a in other_idx:
        product = csi[:, :, a] * np.conj(ref)
        unit = product / (np.abs(product) + eps)
        channels.append(unit.real.astype(np.float32))
        channels.append(unit.imag.astype(np.float32))
    if not channels:
        return np.empty((csi.shape[0], csi.shape[1], 0), dtype=np.float32)
    return np.stack(channels, axis=-1)


def num_feature_channels(num_antennas: int) -> int:
    """Channel count extract_features() produces for a given antenna count:
    NumAntennas amplitude channels + (NumAntennas-1)*2 phase-diff channels."""
    return num_antennas + max(num_antennas - 1, 0) * 2


def extract_features(csi: np.ndarray, reference_antenna: int = 0, amplitude_norm: str = "zscore") -> np.ndarray:
    """Normalized amplitude + cross-antenna phase-difference features, combined.

    Args:
        csi: complex array of shape (T, NumSubcarriers, NumAntennas).
        amplitude_norm: "zscore" (normalize(), per-trial mean/std -- the
            original behavior) or "energy" (energy_normalize(), per-FRAME
            energy -- see that function's docstring for why this is the
            AGC-appropriate granularity and "zscore" structurally can't
            generalize presence detection across sessions).
    Returns:
        float32 array of shape (T, NumSubcarriers, num_feature_channels(NumAntennas)):
        NumAntennas amplitude channels, followed by (NumAntennas-1)*2
        phase-diff (sin, cos) channels. The phase channels are left unscaled
        (not z-scored) -- they're already unit-circle values in [-1, 1], unlike
        amplitude whose raw scale varies a lot per subcarrier.
    """
    if amplitude_norm == "zscore":
        amplitude = normalize(csi)
    elif amplitude_norm == "energy":
        amplitude = energy_normalize(csi, freq_axis=1)
    else:
        raise ValueError(f"unknown amplitude_norm: {amplitude_norm!r}")
    phase = phase_diff_features(csi, reference_antenna)
    return np.concatenate([amplitude, phase], axis=-1)


def sliding_windows(csi: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Segment a (T, ...) time series into overlapping windows.

    Returns an array of shape (NumWindows, window_size, ...).
    """
    t = csi.shape[0]
    if t < window_size:
        raise ValueError(f"Sequence length {t} is shorter than window_size {window_size}")
    starts = range(0, t - window_size + 1, stride)
    return np.stack([csi[s : s + window_size] for s in starts], axis=0)


def _fill_and_smooth(positions: np.ndarray, smoothing: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate NaN gaps (nothing detected in the video at that
    time), then lightly smooth, shared by speed_labels and position_labels."""
    x, y = positions[:, 0].copy(), positions[:, 1].copy()
    idx = np.arange(len(x))
    valid = ~np.isnan(x)
    if valid.sum() >= 2:
        x = np.interp(idx, idx[valid], x[valid])
        y = np.interp(idx, idx[valid], y[valid])
    else:
        # nothing usable in this trial (e.g. empty-room trial with a stray
        # false-positive blob or two) -- treat as stationary throughout
        x[:] = np.nanmean(x) if valid.any() else 0.0
        y[:] = np.nanmean(y) if valid.any() else 0.0

    if smoothing > 1:
        kernel = np.ones(smoothing) / smoothing
        x = np.convolve(x, kernel, mode="same")
        y = np.convolve(y, kernel, mode="same")
    return x, y


def speed_labels(positions: np.ndarray, window_size: int, stride: int, smoothing: int = 5) -> np.ndarray:
    """Whole-window aggregate regression target: mean frame-to-frame
    displacement within each window (Eq. 2 in the paper draft) -- the
    no-grid-binning counterpart to grid_motion_labels below, which bins
    the same per-step displacement by cell instead of averaging it over
    the whole window. Reconstructed from the paper's formula and
    grid_motion_labels' own docstring (which references this function's
    convention) -- the original definition was refactored away when
    grid_motion_labels was added, orphaning train_regression.py, which
    still imports it (see _removed_legacy/training/train_regression.py).

    Uses only strictly in-window frame-to-frame displacements (window_size-1
    steps per window), same convention as grid_motion_labels.

    Returns (NumWindows,) float32.
    """
    x, y = _fill_and_smooth(positions, smoothing)
    step_dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)  # (T-1,)

    t = len(positions)
    starts = list(range(0, t - window_size + 1, stride))
    out = np.zeros(len(starts), dtype=np.float32)
    for wi, s in enumerate(starts):
        end = min(s + window_size - 1, t - 1)
        out[wi] = step_dist[s:end].mean() if end > s else 0.0
    return out


def grid_motion_labels(
    positions: np.ndarray,
    bounds: tuple[float, float, float, float],
    grid_dim: int,
    window_size: int,
    stride: int,
    smoothing: int = 5,
) -> np.ndarray:
    """Coarse spatial regression target: bin the floor into a grid_dim x
    grid_dim grid and, per window, sum the in-window frame-to-frame
    displacement whose starting frame falls in each cell -- a
    (grid_dim**2,)-dim vector per window, mostly zero, answering "how much
    motion happened, and roughly where" rather than a single averaged
    position (see conversation notes: plain continuous (x,y) regression
    scored weakly, R^2=0.158, on a diagnostic elsewhere).

    `bounds` = (x_min, x_max, y_min, y_max) should be computed once across
    every trial being used (not per trial), so a given cell index means the
    same physical region regardless of which trial a window came from.
    Still in the same (uncalibrated) pixel space as `positions` -- cells
    are not equal-area on the real floor plan (see calibrate_homography.py,
    not yet run); fine for training/feasibility, not for a calibrated
    final result.

    Uses only strictly in-window frame-to-frame displacements (window_size-1
    steps per window), same convention as speed_labels.

    Returns (NumWindows, grid_dim**2) float32.
    """
    x, y = _fill_and_smooth(positions, smoothing)
    x_min, x_max, y_min, y_max = bounds
    col = np.clip(((x - x_min) / (x_max - x_min + 1e-8) * grid_dim).astype(int), 0, grid_dim - 1)
    row = np.clip(((y - y_min) / (y_max - y_min + 1e-8) * grid_dim).astype(int), 0, grid_dim - 1)
    cells = row * grid_dim + col

    step_dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    step_cell = cells[:-1]

    t = len(positions)
    starts = range(0, t - window_size + 1, stride)
    k = grid_dim * grid_dim
    out = np.zeros((len(starts), k), dtype=np.float32)
    for wi, s in enumerate(starts):
        end = min(s + window_size - 1, t - 1)
        np.add.at(out[wi], step_cell[s:end], step_dist[s:end])
    return out
