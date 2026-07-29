"""Builds the windowed presence+position dataset from the raw captures:
CSI amplitude (z-scored per subcarrier/antenna/trial) windowed against
resampled head-position trajectories. Produces two labels per window:
presence (was the person visible for >=MIN_VALID_FRAC of the window) and
position (mean head xy, only meaningful when present).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from snn_csi_tracking.data.preprocessing import denoise_amplitude as _denoise_amplitude
from snn_csi_tracking.data.preprocessing import extract_features as _extract_features_tsa
from snn_csi_tracking.data.preprocessing import phase_diff_features
from snn_csi_tracking.data.raw_capture_loader import (
    Capture,
    discover_captures,
    load_empty_room_baseline,
    load_empty_room_cir_baseline,
    parse_csi_file,
)
from snn_csi_tracking.data.trajectory_extraction import (
    classify_presence_from_gaps,
    load_or_extract_trajectory,
    setup_pose_landmarker,
)

MIN_VALID_FRAC = 0.5


def _feature_tag(
    use_phase: bool, use_cir: bool = False, denoise: str | None = None, use_motion_magnitude: bool = False,
    amplitude_norm: str = "zscore", use_relative_motion: bool = False,
    use_spectral_ratio: bool = False, use_cross_coherence: bool = False, use_baseline_deviation: bool = False,
) -> str:
    tag = "amp" if not use_phase else "ampphase"
    tag = f"{tag}cir" if use_cir else tag
    tag = f"{tag}mot" if use_motion_magnitude else tag
    tag = f"{tag}relmot" if use_relative_motion else tag
    tag = f"{tag}specratio" if use_spectral_ratio else tag
    tag = f"{tag}coh" if use_cross_coherence else tag
    tag = f"{tag}basedev" if use_baseline_deviation else tag
    tag = f"{tag}_enorm" if amplitude_norm == "energy" else tag
    return f"{tag}_{denoise}denoise" if denoise else tag


def _denoise(amp: np.ndarray, denoise: str | None) -> np.ndarray:
    """amp: (NumAntennas, NumSubcarriers, T), this pipeline's convention."""
    return _denoise_amplitude(amp, denoise, freq_axis=1, time_axis=2)


def motion_magnitude_feature(csi: np.ndarray) -> np.ndarray:
    """csi: (NumAntennas, NumSubcarriers, T) complex. Returns (1, NumSubcarriers,
    T) float32: mean absolute frame-to-frame amplitude change across every
    antenna/subcarrier, broadcast across the subcarrier axis -- a single
    compact "how much is changing right now" scalar per frame, not a
    per-subcarrier delta channel (see conversation: a full per-subcarrier
    delta-rate companion channel doubled the whole feature width and
    overfit badly; this is the same underlying idea at a dimensionality
    that doesn't blow up the first layer's parameter count)."""
    amp = np.abs(csi)
    diff = np.diff(amp, axis=2, prepend=amp[:, :, :1])
    magnitude = np.abs(diff).mean(axis=(0, 1))  # (T,)
    return np.broadcast_to(magnitude, (csi.shape[1], len(magnitude))).astype(np.float32)[None]


def relative_motion_magnitude_feature(csi: np.ndarray, rolling_frames: int = 20) -> np.ndarray:
    """Like motion_magnitude_feature, but divided by a CAUSAL rolling median
    of its own recent history (past `rolling_frames`, ~1s at 50ms) instead
    of returned as a raw absolute delta.

    Motivation (see conversation): z-scoring is already mathematically
    invariant to a single FIXED multiplicative gain difference between
    trials (a constant factor cancels exactly out of (x-mean)/std) -- it
    still failed to generalize across sessions, which means the real
    nuisance is more likely a gain that DRIFTS continuously within a
    recording (consistent with AGC continuously re-adjusting, not just
    resetting once per file) than a single fixed per-trial scale. A ratio
    against the WHOLE trial's own median has the same blind spot z-scoring
    does, for the same reason -- this instead asks "how much bigger is
    THIS frame's change than a typical frame from the last second", which
    adapts locally as any slow gain drift moves, while still preserving
    the frame-to-frame contrast that signals real motion.

    Returns (1, NumSubcarriers, T) float32, broadcast across subcarriers
    like motion_magnitude_feature."""
    import pandas as pd

    amp = np.abs(csi)
    diff = np.diff(amp, axis=2, prepend=amp[:, :, :1])
    magnitude = np.abs(diff).mean(axis=(0, 1))  # (T,)
    reference = pd.Series(magnitude).rolling(rolling_frames, min_periods=1).median().to_numpy()
    relative = magnitude / (reference + 1e-6)
    return np.broadcast_to(relative, (csi.shape[1], len(relative))).astype(np.float32)[None]


def spectral_motion_ratio_feature(
    csi: np.ndarray, window: int = 64, low_band: tuple[float, float] = (0.15, 3.0),
    high_band: tuple[float, float] = (5.0, 10.0), rate_hz: float = 20.0,
) -> np.ndarray:
    """csi: (NumAntennas, NumSubcarriers, T) complex. Returns (1, NumSubcarriers,
    T) float32: ratio of low-frequency ("human motion band") to high-
    frequency ("noise reference band") power in the amplitude's OWN causal,
    trailing-window temporal spectrum -- see conversation.

    Unlike raw or relative motion magnitude (both about HOW MUCH the signal
    changes), this targets HOW it changes: real human motion (footsteps
    ~1-2Hz, breathing ~0.2-0.3Hz, general sway lower still) concentrates
    energy in a specific low-frequency band, while sensor/AGC noise is
    close to spectrally flat (white) regardless of its absolute loudness.
    A session with a louder-than-usual noise floor raises BOTH the low-band
    and high-band power roughly together, so the ratio stays closer to
    session-invariant than either magnitude-based feature -- the qualitative
    "does this look like a moving body or like flat noise" question, not
    "how big is it."

    Default rate_hz=20 (RATE_MS=50) gives a 10Hz Nyquist limit and ~0.31Hz
    frequency resolution at window=64 (3.2s) -- enough to resolve walking-
    band motion; reliably resolving breathing (~0.2Hz) would need a longer
    window (closer to 8-10s) than this pipeline's current T_WIN.
    """
    amp = np.abs(csi).mean(axis=(0, 1))  # (T,) -- averaged across antenna/subcarrier, same convention as motion_magnitude_feature
    t = len(amp)
    hann = np.hanning(window)
    freqs = np.fft.rfftfreq(window, d=1.0 / rate_hz)
    low_mask = (freqs >= low_band[0]) & (freqs <= low_band[1])
    high_mask = (freqs >= high_band[0]) & (freqs <= high_band[1])

    ratio = np.ones(t, dtype=np.float32)
    for i in range(t):
        start = max(0, i - window + 1)
        seg = amp[start : i + 1]
        if len(seg) < window:
            seg = np.pad(seg, (window - len(seg), 0), mode="edge")
        spec = np.abs(np.fft.rfft(seg * hann)) ** 2
        low_power = spec[low_mask].sum()
        high_power = spec[high_mask].sum()
        ratio[i] = low_power / (high_power + 1e-6)
    # log-scale (dB-like) -- the raw ratio can span many orders of magnitude,
    # which would dominate the other O(1)-scale channels (energy-normalized
    # amplitude, phase in [-1,1]); log brings it to a comparable range.
    log_ratio = np.log10(ratio + 1e-6).astype(np.float32)
    return np.broadcast_to(log_ratio, (csi.shape[1], t)).astype(np.float32)[None]


def cross_antenna_coherence_feature(
    csi: np.ndarray, window: int = 128, sub_seg: int = 32, sub_stride: int = 16,
    band: tuple[float, float] = (0.15, 3.0), rate_hz: float = 20.0,
) -> np.ndarray:
    """csi: (NumAntennas, NumSubcarriers, T) complex. Returns (1,
    NumSubcarriers, T) float32: magnitude-squared coherence between
    antenna pairs, averaged within `band`, computed causally over a
    trailing `window`-frame stretch -- see conversation.

    A genuinely NEW axis from spectral_motion_ratio_feature (which asks
    "does THIS ONE signal look structured over time"): this asks "do
    MULTIPLE INDEPENDENT antennas agree with each other." A real moving
    body reflects signal that reaches every antenna from the same physical
    source, so real motion should show correlated variation across
    antennas at the same frequencies. Each antenna's own thermal/hardware
    noise is an independent physical process -- antenna 0's noise has no
    reason to correlate with antenna 1's noise. So high cross-antenna
    coherence in the motion band indicates a real physical event; low
    coherence indicates each antenna is just doing its own independent
    noise thing, regardless of how much energy either alone shows there.

    IMPORTANT: coherence from a SINGLE FFT snapshot is trivially always 1.0
    by construction (a complex number divided by its own magnitude) -- a
    meaningful estimate needs averaging the cross/auto spectra over
    multiple sub-segments first (Welch's method), which is why this takes
    a longer outer `window` (128 frames/6.4s) divided into overlapping
    `sub_seg`-length pieces, rather than one FFT of the whole window.
    """
    amp = np.abs(csi).mean(axis=1)  # (NumAntennas, T) -- averaged across subcarriers, antenna axis kept
    num_antennas, t = amp.shape
    hann = np.hanning(sub_seg)
    freqs = np.fft.rfftfreq(sub_seg, d=1.0 / rate_hz)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    pairs = [(a, b) for a in range(num_antennas) for b in range(a + 1, num_antennas)]
    sub_starts = list(range(0, window - sub_seg + 1, sub_stride))

    coherence = np.zeros(t, dtype=np.float32)
    for i in range(t):
        start = max(0, i - window + 1)
        seg = amp[:, start : i + 1]
        if seg.shape[1] < window:
            seg = np.pad(seg, ((0, 0), (window - seg.shape[1], 0)), mode="edge")

        specs = np.stack(
            [np.fft.rfft(seg[:, s : s + sub_seg] * hann[None, :], axis=1) for s in sub_starts], axis=0
        )  # (num_subsegs, NumAntennas, freq_bins)
        pair_coh = []
        for a, b in pairs:
            cross = (specs[:, a] * np.conj(specs[:, b])).mean(axis=0)  # averaged over sub-segments
            auto_a = (np.abs(specs[:, a]) ** 2).mean(axis=0)
            auto_b = (np.abs(specs[:, b]) ** 2).mean(axis=0)
            msc = np.abs(cross) ** 2 / (auto_a * auto_b + 1e-12)
            pair_coh.append(msc[band_mask].mean())
        coherence[i] = np.mean(pair_coh)
    return np.broadcast_to(coherence, (csi.shape[1], t)).astype(np.float32)[None]


def empty_baseline_deviation_feature(
    csi: np.ndarray, baseline: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """csi: (NumAntennas, NumSubcarriers, T) complex. baseline: (mean, std),
    each (NumAntennas, NumSubcarriers, 1), from raw_capture_loader.
    load_empty_room_baseline. Returns (NumAntennas, NumSubcarriers, T)
    float32: this trial's raw amplitude z-scored against the condition's
    person-free empty-room reference -- see conversation.

    Unlike every other feature in this module (relative_motion,
    spectral_ratio, cross_coherence), which all measure HOW the signal
    changes over time, this is a STATIC signal: a body sitting motionless
    still perturbs the channel relative to the true empty-room profile, so
    this stays large for as long as someone is physically present, even
    through a stretch with near-zero frame-to-frame change -- directly
    targeting the presence-flicker-during-sitting failure mode observed on
    the L activity's spy-overlay video (person sits, ground truth says
    present, every motion-based feature goes quiet).

    Added as an EXTRA channel (see compute_features' use_baseline_deviation
    flag), alongside whatever per-trial-normalized amplitude representation
    is already in use -- NOT a replacement for it. An earlier experiment
    used this same empty-room reference as the ONLY amplitude
    representation (compute_features' `baseline` replace-mode) and was
    dropped for cross-session gain drift; stacking it as one additional
    channel lets the model fall back on the per-trial-normalized channels
    when this one is unreliable, instead of depending on it exclusively.
    """
    mu, sigma = baseline
    amp = np.abs(csi)
    return ((amp - mu) / (sigma + 1e-8)).astype(np.float32)


def compute_features(
    csi: np.ndarray,
    use_phase: bool = True,
    baseline: tuple[np.ndarray, np.ndarray] | None = None,
    use_cir: bool = False,
    cir_baseline: tuple[np.ndarray, np.ndarray] | None = None,
    denoise: str | None = None,
    use_motion_magnitude: bool = False,
    amplitude_norm: str = "zscore",
    use_relative_motion: bool = False,
    use_spectral_ratio: bool = False,
    use_cross_coherence: bool = False,
    baseline_deviation: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """csi: (NumAntennas, NumSubcarriers, T) complex, this pipeline's convention.
    Returns (NumChannels, NumSubcarriers, T) float32.

    use_phase=True: z-scored amplitude per antenna + cross-antenna
    phase-difference (sin, cos) channels, via preprocessing.extract_features()
    -- reused as-is (already validated by the grid-motion pipeline) rather
    than re-deriving the phase math. That function expects/returns the
    (T, Sub, Ant/Chan) convention, so this only transposes in and back out.
    use_phase=False: z-scored amplitude only (this pipeline's original
    behavior) -- kept for a direct before/after comparison.

    baseline: optional (mean, std) each (NumAntennas, NumSubcarriers, 1),
    from raw_capture_loader.load_empty_room_baseline -- a person-free
    reference profile to z-score amplitude against instead of this trial's
    own mean/std across time. The trial's own mean/std is contaminated by
    whatever frames the person actually occupies (biasing the "static"
    estimate toward however much they moved during this one trial); the
    empty-room profile is a genuinely person-free static-clutter estimate,
    shared across every trial of the same condition. Only affects the
    amplitude channels -- phase-difference channels already cancel shared
    CFO/SFO drift via the conjugate-product trick and don't need one.

    use_cir=True: adds NumAntennas more channels, the magnitude of the
    channel impulse response (IFFT across the subcarrier/frequency axis,
    per antenna) -- unlike every feature above, which stays in the
    frequency domain where every subcarrier's value is a sum over every
    multipath reflection's contribution at that frequency, the delay
    domain separates reflections by path length into distinct taps. Full-
    length IFFT (same tap count as input subcarriers), z-scored the same
    way as amplitude -- against `cir_baseline` (raw_capture_loader.
    load_empty_room_cir_baseline) if given, else this trial's own mean/std.

    denoise: None, "wavelet", or "pca" (see preprocessing.wavelet_denoise /
    pca_denoise) -- applied to raw amplitude before z-scoring, on the
    theory that most CSI-based position/localization work in the
    literature denoises first (see conversation). Only wired into the
    baseline (empty-room-normalized) and amplitude-only paths for now, not
    the plain per-trial-normalized use_phase path.

    use_motion_magnitude=True: adds 1 more channel, see
    motion_magnitude_feature. use_relative_motion=True: uses
    relative_motion_magnitude_feature instead (rolling-median-normalized,
    see that function's docstring) -- mutually exclusive with
    use_motion_magnitude in practice, but both flags are independent so
    callers control which one adds the extra channel.

    amplitude_norm: "zscore" (per-trial mean/std, the original behavior) or
    "energy" (per-FRAME energy normalization, preprocessing.energy_normalize)
    -- only applies to the no-baseline amplitude path (baseline=None); see
    that function's docstring for why "zscore" structurally can't
    generalize presence detection across sessions (confirmed empirically:
    near-identical raw output on two held-out sessions with opposite
    ground truth) and "energy" is the AGC-appropriate fix.
    """
    if not use_phase:
        amp = _denoise(np.abs(csi), denoise)
        if baseline is not None:
            mu, sigma = baseline
            return ((amp - mu) / (sigma + 1e-8)).astype(np.float32)
        if amplitude_norm == "energy":
            energy = np.sqrt((amp ** 2).sum(axis=1, keepdims=True))
            return (amp / (energy + 1e-8)).astype(np.float32)
        mu, sigma = amp.mean(axis=2, keepdims=True), amp.std(axis=2, keepdims=True)
        return ((amp - mu) / (sigma + 1e-8)).astype(np.float32)

    if baseline is not None:
        amp = _denoise(np.abs(csi), denoise)
        mu, sigma = baseline
        amp_z = ((amp - mu) / (sigma + 1e-8)).astype(np.float32)
        phase = phase_diff_features(csi.transpose(2, 1, 0)).transpose(2, 1, 0).astype(np.float32)
        feat = np.concatenate([amp_z, phase], axis=0)
    else:
        csi_tsa = csi.transpose(2, 1, 0)  # (T, Sub, Ant)
        feat_tsa = _extract_features_tsa(csi_tsa, amplitude_norm=amplitude_norm)  # (T, Sub, Chan)
        feat = feat_tsa.transpose(2, 1, 0).astype(np.float32)  # (Chan, Sub, T)
    if use_cir:
        cir_mag = np.abs(np.fft.ifft(csi, axis=1)).astype(np.float32)
        if cir_baseline is not None:
            mu, sigma = cir_baseline
        else:
            mu, sigma = cir_mag.mean(axis=2, keepdims=True), cir_mag.std(axis=2, keepdims=True)
        cir_feat = ((cir_mag - mu) / (sigma + 1e-8)).astype(np.float32)
        feat = np.concatenate([feat, cir_feat], axis=0)
    if use_motion_magnitude:
        feat = np.concatenate([feat, motion_magnitude_feature(csi)], axis=0)
    if use_relative_motion:
        feat = np.concatenate([feat, relative_motion_magnitude_feature(csi)], axis=0)
    if use_spectral_ratio:
        feat = np.concatenate([feat, spectral_motion_ratio_feature(csi)], axis=0)
    if use_cross_coherence:
        feat = np.concatenate([feat, cross_antenna_coherence_feature(csi)], axis=0)
    if baseline_deviation is not None:
        feat = np.concatenate([feat, empty_baseline_deviation_feature(csi, baseline_deviation)], axis=0)
    return feat


def _resolve_trajectory(capture: Capture, trajectory_cache_dir: Path, get_landmarker, frame_skip: int) -> np.ndarray | None:
    """Shared by build_dataset/build_perframe_dataset: cache hit avoids
    ever touching MediaPipe (most trials only carry the cache forward, see
    discover_captures())."""
    cache_path = trajectory_cache_dir / f"{capture.trial_key}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    landmarker, mp = get_landmarker()
    return load_or_extract_trajectory(
        capture.video_path, capture.trial_key, trajectory_cache_dir, landmarker, mp, frame_skip=frame_skip
    )


def resample_depth(depth_raw: np.ndarray, n_target: int) -> np.ndarray:
    """depth_raw: raw per-trajectory-frame metric depth in meters (see
    depth_extraction.py's metric-indoor checkpoint). Resamples to n_target
    (the CSI frame count), matching (x, y)'s own resampling convention.

    Deliberately does NOT min-max normalize per trial (the previous
    resample_and_normalize_depth did) -- that would destroy the metric
    scale this whole pipeline exists to produce, and would make the same
    real depth map to a different number in every trial depending on that
    trial's own min/max range. Shared by the dataset builder and any script
    that needs to reconstruct the same target z a 3D checkpoint was trained
    against (e.g. visualize_trajectory_prediction.py)."""
    return np.interp(np.linspace(0, len(depth_raw) - 1, n_target), np.arange(len(depth_raw)), depth_raw)


def resample_to_n(traj: np.ndarray, n_target: int) -> np.ndarray:
    n_src = len(traj)
    t_src = np.linspace(0, 1, n_src)
    t_tgt = np.linspace(0, 1, n_target)
    return np.stack(
        [np.interp(t_tgt, t_src, traj[:, 0]), np.interp(t_tgt, t_src, traj[:, 1])], axis=1
    )


def make_position_windows(
    amp_z: np.ndarray, positions: np.ndarray, t_win: int, stride: int, min_valid_frac: float = MIN_VALID_FRAC
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """amp_z: (NumAntennas, NumSubcarriers, N). positions: (N, 2).
    Returns a list of (window, target_xy, presence_flag)."""
    n = amp_z.shape[2]
    out = []
    for start in range(0, n - t_win + 1, stride):
        end = start + t_win
        p = positions[start:end]
        valid_frac = 1 - np.isnan(p[:, 0]).mean()
        present = valid_frac >= min_valid_frac
        target_xy = np.nanmean(p, axis=0) if present else np.array([0.5, 0.5])
        out.append((amp_z[:, :, start:end], target_xy.astype(np.float32), float(present)))
    return out


def build_dataset(
    raw_root: Path,
    trajectory_cache_dir: Path,
    rate_ms: int = 50,
    t_win: int = 64,
    stride: int = 32,
    frame_skip: int = 2,
    use_phase: bool = True,
    use_empty_baseline: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, present, groups, activity_codes):
    X: (NumWindows, NumChannels, NumSubcarriers, t_win) float32 -- see
        compute_features for what NumChannels is (3, 7, or 9)
    y: (NumWindows, 2) float32 -- mean head (x, y), meaningful only where present=1
    present: (NumWindows,) float32 -- 1.0 if the person was visible >=MIN_VALID_FRAC of the window
    groups: (NumWindows,) str -- trial key, for a chronological per-trial split
    activity_codes: (NumWindows,) str

    use_empty_baseline: z-score amplitude against each condition's genuine
    empty-room profile (raw_capture_loader.load_empty_room_baseline) instead
    of each trial's own mean/std -- see compute_features's `baseline` arg.
    """
    captures = discover_captures(raw_root)
    print(f"{len(captures)} raw captures found under {raw_root}")

    baselines = {}
    if use_empty_baseline:
        for condition in ("NLoS", "PLoS"):
            baselines[condition] = load_empty_room_baseline(raw_root, condition)
            status = "found" if baselines[condition] is not None else "MISSING -- falling back to per-trial normalization"
            print(f"empty-room baseline for {condition}: {status}")

    landmarker_state: list = []  # lazy-init: most trials hit the trajectory cache and never
                                  # need MediaPipe (which would otherwise fetch its model file)

    def get_landmarker():
        if not landmarker_state:
            landmarker_state.append(setup_pose_landmarker())
        return landmarker_state[0]

    X_list, y_list, present_list, groups, activity_codes = [], [], [], [], []
    t_start = time.time()
    for i, capture in enumerate(captures, start=1):
        csi, _timestamps = parse_csi_file(capture.csi_path)
        if csi.shape[2] == 0:
            print(f"  WARNING: {capture.trial_key} has no valid CSI frames -- skipping")
            continue

        baseline = baselines.get(capture.condition) if use_empty_baseline else None
        amp_z = compute_features(csi, use_phase=use_phase, baseline=baseline)

        trajectory = _resolve_trajectory(capture, trajectory_cache_dir, get_landmarker, frame_skip)
        if trajectory is None:
            print(f"  WARNING: no cached trajectory or video for {capture.trial_key} -- skipping")
            continue
        positions = resample_to_n(trajectory, csi.shape[2])

        windows = make_position_windows(amp_z, positions, t_win, stride)
        for w, xy, present in windows:
            X_list.append(w)
            y_list.append(xy)
            present_list.append(present)
            groups.append(capture.trial_key)
            activity_codes.append(capture.activity)

        elapsed = time.time() - t_start
        print(f"[{i}/{len(captures)}] {capture.trial_key}: {len(windows)} windows  ({elapsed:.1f}s elapsed)")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0)
    present = np.array(present_list, dtype=np.float32)
    groups = np.array(groups)
    activity_codes = np.array(activity_codes)
    print(f"presence rate: {present.mean():.2f}")
    return X, y, present, groups, activity_codes


def cache_paths(
    cache_dir: Path, rate_ms: int, t_win: int, stride: int,
    use_phase: bool = True, use_empty_baseline: bool = False,
) -> tuple[Path, Path]:
    base_tag = f"presence_position_{_feature_tag(use_phase)}_{rate_ms}ms_w{t_win}_s{stride}"
    tag = f"{base_tag}_ebase" if use_empty_baseline else base_tag
    return cache_dir / f"{tag}_meta.npz", cache_dir / f"{tag}_X.npy"


def load_or_build_dataset(
    raw_root: Path,
    trajectory_cache_dir: Path,
    cache_dir: Path,
    rate_ms: int = 50,
    t_win: int = 64,
    stride: int = 32,
    use_phase: bool = True,
    use_empty_baseline: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    meta_path, x_path = cache_paths(cache_dir, rate_ms, t_win, stride, use_phase, use_empty_baseline)
    if meta_path.exists() and x_path.exists():
        print(f"Loading cached dataset from {x_path}")
        meta = np.load(meta_path, allow_pickle=True)
        X = np.load(x_path, mmap_mode="r")
        return X, meta["y"], meta["present"], meta["groups"], meta["activity_codes"]

    cache_dir.mkdir(parents=True, exist_ok=True)
    X, y, present, groups, activity_codes = build_dataset(
        raw_root, trajectory_cache_dir, rate_ms=rate_ms, t_win=t_win, stride=stride,
        use_phase=use_phase, use_empty_baseline=use_empty_baseline,
    )
    meta_path, x_path = cache_paths(cache_dir, rate_ms, t_win, stride, use_phase, use_empty_baseline)
    np.save(x_path, X)
    np.savez(meta_path, y=y, present=present, groups=groups, activity_codes=activity_codes)
    print(f"Cached dataset to {x_path}")
    return X, y, present, groups, activity_codes


def make_perframe_windows(
    amp_z: np.ndarray, positions: np.ndarray, t_win: int, stride: int, present: np.ndarray | None = None
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Like make_position_windows, but keeps a per-timestep label instead of
    averaging over the window: positions: (N, out_dim), out_dim=2
    (normalized image-plane x,y) or 3 (real camera-frame X,Y,Z in meters,
    via camera_geometry.deproject_pixel_to_camera_frame -- see depth_extraction.py).

    present: optional precomputed whole-capture presence mask (N,) bool/float,
    from trajectory_extraction.classify_presence_from_gaps -- MUST be
    computed on the full capture, not a windowed slice, since classifying a
    gap as "genuine absence" vs "still present, tracking lost" depends on
    whether the gap touches the capture's true start/end (see that
    function's docstring). Falls back to the naive ~isnan(positions) rule
    if not given, for callers that haven't been updated.

    Returns a list of (window, pos_seq (t_win, out_dim), present_seq (t_win,))."""
    n = amp_z.shape[2]
    if present is None:
        present = ~np.isnan(positions[:, 0])
    out = []
    for start in range(0, n - t_win + 1, stride):
        end = start + t_win
        p = positions[start:end]
        present_seq = present[start:end].astype(np.float32)
        p_filled = np.nan_to_num(p, nan=0.5).astype(np.float32)
        out.append((amp_z[:, :, start:end], p_filled, present_seq))
    return out


def build_capture_cache(
    raw_root: Path,
    trajectory_cache_dir: Path,
    cache_dir: Path,
    use_phase: bool = True,
    use_empty_baseline: bool = False,
    denoise: str | None = None,
    use_motion_magnitude: bool = False,
    amplitude_norm: str = "zscore",
    use_relative_motion: bool = False,
    use_spectral_ratio: bool = False,
    use_cross_coherence: bool = False,
    use_baseline_deviation: bool = False,
    frame_skip: int = 2,
) -> dict[str, dict]:
    """Computes and caches PER-CAPTURE feature arrays -- NOT pre-sliced into
    windows -- so a small stride (even stride=1) never requires
    materializing every heavily-overlapping window as its own on-disk copy.

    A stride=1 windowing of this pipeline's ~50 captures at t_win=64 would
    produce ~56,850 windows; storing each as its own (8, 1024, 64) float32
    array is ~108GB on disk -- >2x the 48GB actually free on this machine
    (see conversation) -- despite consecutive windows sharing 63 of 64
    frames. This function instead caches each capture's feature array ONCE
    (~40MB/capture, ~2GB total, independent of whatever stride is used
    later), and windowing happens on demand in SlidingWindowCaptureDataset
    via cheap array slicing -- the standard, memory-correct way to do
    sliding-window datasets: store the underlying continuous signal, and
    generate window VIEWS/slices at request time, not N nearly-identical
    materialized copies.

    Returns {trial_key: {"feat": (Chan,Sub,T) float32, "pos": (T,2) float32
    (NaN where untracked), "present": (T,) float32, "activity": str,
    "condition": str}}, and also caches this dict to
    `cache_dir/capture_cache_<feature_tag>.npz` for reuse.
    """
    tag = _feature_tag(use_phase, denoise=denoise, use_motion_magnitude=use_motion_magnitude,
                        amplitude_norm=amplitude_norm, use_relative_motion=use_relative_motion,
                        use_spectral_ratio=use_spectral_ratio, use_cross_coherence=use_cross_coherence,
                        use_baseline_deviation=use_baseline_deviation)
    tag = f"{tag}_ebase" if use_empty_baseline else tag
    cache_path = cache_dir / f"capture_cache_{tag}.npz"
    if cache_path.exists():
        print(f"Loading per-capture cache from {cache_path}")
        loaded = np.load(cache_path, allow_pickle=True)
        return loaded["data"].item()

    captures = discover_captures(raw_root)
    print(f"{len(captures)} raw captures found under {raw_root} (building per-capture cache)")

    baselines = {}
    if use_empty_baseline:
        for condition in ("NLoS", "PLoS"):
            baselines[condition] = load_empty_room_baseline(raw_root, condition, denoise=denoise)

    baseline_deviations = {}
    if use_baseline_deviation:
        for condition in ("NLoS", "PLoS"):
            baseline_deviations[condition] = load_empty_room_baseline(raw_root, condition, denoise=denoise)
            if baseline_deviations[condition] is None:
                raise ValueError(
                    f"use_baseline_deviation=True but no empty-room ('_E') capture found for "
                    f"condition {condition!r} -- can't build a consistent channel count without it."
                )

    landmarker_state: list = []

    def get_landmarker():
        if not landmarker_state:
            landmarker_state.append(setup_pose_landmarker())
        return landmarker_state[0]

    data: dict[str, dict] = {}
    t_start = time.time()
    for i, capture in enumerate(captures, start=1):
        csi, _timestamps = parse_csi_file(capture.csi_path)
        if csi.shape[2] == 0:
            print(f"  WARNING: {capture.trial_key} has no valid CSI frames -- skipping")
            continue

        baseline = baselines.get(capture.condition) if use_empty_baseline else None
        baseline_deviation = baseline_deviations.get(capture.condition) if use_baseline_deviation else None
        feat = compute_features(
            csi, use_phase=use_phase, baseline=baseline, denoise=denoise, use_motion_magnitude=use_motion_magnitude,
            amplitude_norm=amplitude_norm, use_relative_motion=use_relative_motion, use_spectral_ratio=use_spectral_ratio,
            use_cross_coherence=use_cross_coherence, baseline_deviation=baseline_deviation,
        )
        trajectory = _resolve_trajectory(capture, trajectory_cache_dir, get_landmarker, frame_skip)
        if trajectory is None:
            print(f"  WARNING: no cached trajectory or video for {capture.trial_key} -- skipping")
            continue
        positions = resample_to_n(trajectory, csi.shape[2])
        present = classify_presence_from_gaps(positions).astype(np.float32)

        data[capture.trial_key] = {
            "feat": feat.astype(np.float32), "pos": positions.astype(np.float32), "present": present,
            "activity": capture.activity, "condition": capture.condition,
        }
        print(f"[{i}/{len(captures)}] {capture.trial_key}: T={feat.shape[-1]}  ({time.time()-t_start:.1f}s elapsed)")

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, data=data)
    print(f"Cached per-capture data to {cache_path}")
    return data


def build_perframe_dataset(
    raw_root: Path,
    trajectory_cache_dir: Path,
    depth_cache_dir: Path,
    rate_ms: int = 50,
    t_win: int = 64,
    stride: int = 32,
    use_3d: bool = False,
    frame_skip: int = 2,
    use_phase: bool = True,
    use_empty_baseline: bool = False,
    denoise: str | None = None,
    use_motion_magnitude: bool = False,
    amplitude_norm: str = "zscore",
    use_relative_motion: bool = False,
    use_spectral_ratio: bool = False,
    use_cross_coherence: bool = False,
    use_baseline_deviation: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, pos, present, groups, activity_codes):
    X: (NumWindows, NumChannels, NumSubcarriers, t_win) float32 -- see
        compute_features for what NumChannels is (3, 7, or 9)
    pos: (NumWindows, t_win, out_dim) float32 -- per-timestep position,
        meaningful only where present=1 at that timestep. 2D: (x, y)
        normalized image-plane. 3D (use_3d=True): real (X, Y, Z) in meters,
        in the CAMERA's own frame (see
        camera_geometry.deproject_pixel_to_camera_frame) -- not room/
        floor-plan coordinates, since that needs a separate camera-pose
        calibration this pipeline doesn't do (see that function's
        docstring for why that's a legitimate simplification here, not a
        gap).
    present: (NumWindows, t_win) float32 -- per-timestep presence
    groups, activity_codes: (NumWindows,), as build_dataset.

    denoise: None, "wavelet", or "pca" --
    see compute_features's `denoise` arg. use_motion_magnitude: see
    compute_features / motion_magnitude_feature. amplitude_norm: see
    compute_features / preprocessing.energy_normalize. use_relative_motion:
    see compute_features / relative_motion_magnitude_feature.
    """
    from snn_csi_tracking.data.camera_geometry import deproject_pixel_to_camera_frame
    from snn_csi_tracking.data.depth_extraction import (
        depth_cache_path, get_video_dimensions, load_or_extract_depth, setup_depth_pipeline,
    )

    captures = discover_captures(raw_root)
    print(f"{len(captures)} raw captures found under {raw_root}")

    baselines = {}
    if use_empty_baseline:
        for condition in ("NLoS", "PLoS"):
            baselines[condition] = load_empty_room_baseline(raw_root, condition, denoise=denoise)
            status = "found" if baselines[condition] is not None else "MISSING -- falling back to per-trial normalization"
            print(f"empty-room baseline for {condition} (denoise={denoise}): {status}")

    baseline_deviations = {}
    if use_baseline_deviation:
        for condition in ("NLoS", "PLoS"):
            baseline_deviations[condition] = load_empty_room_baseline(raw_root, condition, denoise=denoise)
            if baseline_deviations[condition] is None:
                # Unlike use_empty_baseline (a normalization CHOICE that falls back
                # cleanly to per-trial norm either way -- same channel count both
                # branches), this ADDS a channel -- a missing baseline for one
                # condition would silently give that condition's windows fewer
                # channels than the other's, breaking the np.stack below.
                raise ValueError(
                    f"use_baseline_deviation=True but no empty-room ('_E') capture found for "
                    f"condition {condition!r} -- can't build a consistent channel count without it."
                )
            print(f"empty-room baseline (for deviation channel) for {condition} (denoise={denoise}): found")

    landmarker_state: list = []
    depth_state: list = []

    def get_landmarker():
        if not landmarker_state:
            landmarker_state.append(setup_pose_landmarker())
        return landmarker_state[0]

    def get_depth_pipe():
        if not depth_state:
            depth_state.append(setup_depth_pipeline())
        return depth_state[0]

    X_list, pos_list, present_list, groups, activity_codes = [], [], [], [], []
    t_start = time.time()
    for i, capture in enumerate(captures, start=1):
        csi, _timestamps = parse_csi_file(capture.csi_path)
        if csi.shape[2] == 0:
            print(f"  WARNING: {capture.trial_key} has no valid CSI frames -- skipping")
            continue

        baseline = baselines.get(capture.condition) if use_empty_baseline else None
        baseline_deviation = baseline_deviations.get(capture.condition) if use_baseline_deviation else None
        amp_z = compute_features(
            csi, use_phase=use_phase, baseline=baseline, denoise=denoise, use_motion_magnitude=use_motion_magnitude,
            amplitude_norm=amplitude_norm, use_relative_motion=use_relative_motion, use_spectral_ratio=use_spectral_ratio,
            use_cross_coherence=use_cross_coherence, baseline_deviation=baseline_deviation,
        )

        trajectory = _resolve_trajectory(capture, trajectory_cache_dir, get_landmarker, frame_skip)
        if trajectory is None:
            print(f"  WARNING: no cached trajectory or video for {capture.trial_key} -- skipping")
            continue
        positions = resample_to_n(trajectory, csi.shape[2])
        present_mask = classify_presence_from_gaps(positions)

        if use_3d:
            cached_depth = depth_cache_path(depth_cache_dir, capture.trial_key).exists()
            depth_raw = load_or_extract_depth(
                capture.video_path, capture.trial_key, trajectory, depth_cache_dir,
                None if cached_depth else get_depth_pipe(),
            )
            if depth_raw is None:
                print(f"  WARNING: no cached depth or video for {capture.trial_key} -- skipping")
                continue
            depth_resampled = resample_depth(depth_raw, csi.shape[2])
            width, height = get_video_dimensions(capture.video_path)
            positions = deproject_pixel_to_camera_frame(positions, depth_resampled, width, height).astype(np.float32)

        windows = make_perframe_windows(amp_z, positions, t_win, stride, present=present_mask)
        for w, pos_seq, present_seq in windows:
            X_list.append(w)
            pos_list.append(pos_seq)
            present_list.append(present_seq)
            groups.append(capture.trial_key)
            activity_codes.append(capture.activity)

        elapsed = time.time() - t_start
        print(f"[{i}/{len(captures)}] {capture.trial_key}: {len(windows)} windows  ({elapsed:.1f}s elapsed)")

    X = np.stack(X_list, axis=0).astype(np.float32)
    pos = np.stack(pos_list, axis=0)
    present = np.stack(present_list, axis=0)
    groups = np.array(groups)
    activity_codes = np.array(activity_codes)
    print(f"presence rate: {present.mean():.2f}")
    return X, pos, present, groups, activity_codes


def perframe_cache_paths(
    cache_dir: Path, rate_ms: int, t_win: int, stride: int, use_3d: bool,
    use_phase: bool = True, use_empty_baseline: bool = False,
    denoise: str | None = None, use_motion_magnitude: bool = False, amplitude_norm: str = "zscore",
    use_relative_motion: bool = False, use_spectral_ratio: bool = False, use_cross_coherence: bool = False,
    use_baseline_deviation: bool = False,
) -> tuple[Path, Path]:
    dim_tag = "3d" if use_3d else "2d"
    base_tag = f"presence_position_perframe_{_feature_tag(use_phase, denoise=denoise, use_motion_magnitude=use_motion_magnitude, amplitude_norm=amplitude_norm, use_relative_motion=use_relative_motion, use_spectral_ratio=use_spectral_ratio, use_cross_coherence=use_cross_coherence, use_baseline_deviation=use_baseline_deviation)}_{dim_tag}_{rate_ms}ms_w{t_win}_s{stride}"
    tag = f"{base_tag}_ebase" if use_empty_baseline else base_tag
    return cache_dir / f"{tag}_meta.npz", cache_dir / f"{tag}_X.npy"


def load_or_build_perframe_dataset(
    raw_root: Path,
    trajectory_cache_dir: Path,
    depth_cache_dir: Path,
    cache_dir: Path,
    rate_ms: int = 50,
    t_win: int = 64,
    stride: int = 32,
    use_3d: bool = False,
    use_phase: bool = True,
    use_empty_baseline: bool = False,
    denoise: str | None = None,
    use_motion_magnitude: bool = False,
    amplitude_norm: str = "zscore",
    use_relative_motion: bool = False,
    use_spectral_ratio: bool = False,
    use_cross_coherence: bool = False,
    use_baseline_deviation: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    meta_path, x_path = perframe_cache_paths(cache_dir, rate_ms, t_win, stride, use_3d, use_phase, use_empty_baseline, denoise, use_motion_magnitude, amplitude_norm, use_relative_motion, use_spectral_ratio, use_cross_coherence, use_baseline_deviation)
    if meta_path.exists() and x_path.exists():
        print(f"Loading cached dataset from {x_path}")
        meta = np.load(meta_path, allow_pickle=True)
        X = np.load(x_path, mmap_mode="r")
        return X, meta["pos"], meta["present"], meta["groups"], meta["activity_codes"]

    cache_dir.mkdir(parents=True, exist_ok=True)
    X, pos, present, groups, activity_codes = build_perframe_dataset(
        raw_root, trajectory_cache_dir, depth_cache_dir, rate_ms=rate_ms, t_win=t_win, stride=stride,
        use_3d=use_3d, use_phase=use_phase, use_empty_baseline=use_empty_baseline, denoise=denoise,
        use_motion_magnitude=use_motion_magnitude, amplitude_norm=amplitude_norm,
        use_relative_motion=use_relative_motion, use_spectral_ratio=use_spectral_ratio,
        use_cross_coherence=use_cross_coherence, use_baseline_deviation=use_baseline_deviation,
    )
    meta_path, x_path = perframe_cache_paths(cache_dir, rate_ms, t_win, stride, use_3d, use_phase, use_empty_baseline, denoise, use_motion_magnitude, amplitude_norm, use_relative_motion, use_spectral_ratio, use_cross_coherence, use_baseline_deviation)
    np.save(x_path, X)
    np.savez(meta_path, pos=pos, present=present, groups=groups, activity_codes=activity_codes)
    print(f"Cached dataset to {x_path}")
    return X, pos, present, groups, activity_codes


def make_perframe_grid_windows(
    amp_z: np.ndarray, positions: np.ndarray, grid_dim: int, t_win: int, stride: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Like make_perframe_windows, but the per-timestep label is a discrete
    grid-cell class index (grid_dim**2 classes) instead of continuous (x, y)
    -- a coarser, more forgiving spatial target (see preprocessing.
    grid_motion_labels: plain continuous position regression scored weakly,
    R^2=0.158, on a diagnostic elsewhere; the fix that actually worked there
    was reformulating the target, not a better model).

    `positions` is already in this pipeline's normalized [0, 1] image-frame
    coordinates (see trajectory_extraction.extract_head_trajectory), unlike
    grid_motion_labels' raw, uncalibrated pixel space -- so cell bounds are
    just the fixed unit square, no per-dataset bounds computation needed.

    Returns a list of (window, cell_seq (t_win,) int64, present_seq (t_win,)).
    Cell indices at absent/invalid frames are meaningless (0) -- always mask
    by present_seq, same convention as make_perframe_windows' filled position.
    """
    n = amp_z.shape[2]
    x = np.nan_to_num(positions[:, 0], nan=0.0)
    y = np.nan_to_num(positions[:, 1], nan=0.0)
    col = np.clip((x * grid_dim).astype(np.int64), 0, grid_dim - 1)
    row = np.clip((y * grid_dim).astype(np.int64), 0, grid_dim - 1)
    cells = row * grid_dim + col

    out = []
    for start in range(0, n - t_win + 1, stride):
        end = start + t_win
        present_seq = (~np.isnan(positions[start:end, 0])).astype(np.float32)
        cell_seq = cells[start:end]
        out.append((amp_z[:, :, start:end], cell_seq, present_seq))
    return out


def build_perframe_grid_dataset(
    raw_root: Path,
    trajectory_cache_dir: Path,
    grid_dim: int,
    rate_ms: int = 50,
    t_win: int = 64,
    stride: int = 32,
    frame_skip: int = 2,
    use_phase: bool = True,
    use_empty_baseline: bool = False,
    use_cir: bool = False,
    denoise: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2D-only grid-cell counterpart to build_perframe_dataset. Returns
    (X, cells, present, groups, activity_codes):
    X: (NumWindows, NumChannels, NumSubcarriers, t_win) float32
    cells: (NumWindows, t_win) int64 -- per-timestep grid-cell class index
    present: (NumWindows, t_win) float32 -- per-timestep presence
    groups, activity_codes: (NumWindows,), as build_dataset.
    """
    captures = discover_captures(raw_root)
    print(f"{len(captures)} raw captures found under {raw_root}")

    baselines, cir_baselines = {}, {}
    if use_empty_baseline:
        for condition in ("NLoS", "PLoS"):
            baselines[condition] = load_empty_room_baseline(raw_root, condition, denoise=denoise)
            status = "found" if baselines[condition] is not None else "MISSING -- falling back to per-trial normalization"
            print(f"empty-room baseline for {condition} (denoise={denoise}): {status}")
            if use_cir:
                cir_baselines[condition] = load_empty_room_cir_baseline(raw_root, condition)
                status = "found" if cir_baselines[condition] is not None else "MISSING -- falling back to per-trial normalization"
                print(f"empty-room CIR baseline for {condition}: {status}")

    landmarker_state: list = []

    def get_landmarker():
        if not landmarker_state:
            landmarker_state.append(setup_pose_landmarker())
        return landmarker_state[0]

    X_list, cell_list, present_list, groups, activity_codes = [], [], [], [], []
    t_start = time.time()
    for i, capture in enumerate(captures, start=1):
        csi, _timestamps = parse_csi_file(capture.csi_path)
        if csi.shape[2] == 0:
            print(f"  WARNING: {capture.trial_key} has no valid CSI frames -- skipping")
            continue

        baseline = baselines.get(capture.condition) if use_empty_baseline else None
        cir_baseline = cir_baselines.get(capture.condition) if (use_empty_baseline and use_cir) else None
        amp_z = compute_features(
            csi, use_phase=use_phase, baseline=baseline,
            use_cir=use_cir, cir_baseline=cir_baseline, denoise=denoise,
        )

        trajectory = _resolve_trajectory(capture, trajectory_cache_dir, get_landmarker, frame_skip)
        if trajectory is None:
            print(f"  WARNING: no cached trajectory or video for {capture.trial_key} -- skipping")
            continue
        positions = resample_to_n(trajectory, csi.shape[2])

        windows = make_perframe_grid_windows(amp_z, positions, grid_dim, t_win, stride)
        for w, cell_seq, present_seq in windows:
            X_list.append(w)
            cell_list.append(cell_seq)
            present_list.append(present_seq)
            groups.append(capture.trial_key)
            activity_codes.append(capture.activity)

        elapsed = time.time() - t_start
        print(f"[{i}/{len(captures)}] {capture.trial_key}: {len(windows)} windows  ({elapsed:.1f}s elapsed)")

    X = np.stack(X_list, axis=0).astype(np.float32)
    cells = np.stack(cell_list, axis=0)
    present = np.stack(present_list, axis=0)
    groups = np.array(groups)
    activity_codes = np.array(activity_codes)
    print(f"presence rate: {present.mean():.2f}")
    return X, cells, present, groups, activity_codes


def perframe_grid_cache_paths(
    cache_dir: Path, rate_ms: int, t_win: int, stride: int, grid_dim: int,
    use_phase: bool = True, use_empty_baseline: bool = False, use_cir: bool = False,
    denoise: str | None = None,
) -> tuple[Path, Path]:
    base_tag = f"presence_position_perframe_grid{grid_dim}_{_feature_tag(use_phase, use_cir, denoise)}_{rate_ms}ms_w{t_win}_s{stride}"
    tag = f"{base_tag}_ebase" if use_empty_baseline else base_tag
    return cache_dir / f"{tag}_meta.npz", cache_dir / f"{tag}_X.npy"


def load_or_build_perframe_grid_dataset(
    raw_root: Path,
    trajectory_cache_dir: Path,
    cache_dir: Path,
    grid_dim: int,
    rate_ms: int = 50,
    t_win: int = 64,
    stride: int = 32,
    use_phase: bool = True,
    use_empty_baseline: bool = False,
    use_cir: bool = False,
    denoise: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    meta_path, x_path = perframe_grid_cache_paths(cache_dir, rate_ms, t_win, stride, grid_dim, use_phase, use_empty_baseline, use_cir, denoise)
    if meta_path.exists() and x_path.exists():
        print(f"Loading cached dataset from {x_path}")
        meta = np.load(meta_path, allow_pickle=True)
        X = np.load(x_path, mmap_mode="r")
        return X, meta["cells"], meta["present"], meta["groups"], meta["activity_codes"]

    cache_dir.mkdir(parents=True, exist_ok=True)
    X, cells, present, groups, activity_codes = build_perframe_grid_dataset(
        raw_root, trajectory_cache_dir, grid_dim, rate_ms=rate_ms, t_win=t_win, stride=stride,
        use_phase=use_phase, use_empty_baseline=use_empty_baseline, use_cir=use_cir, denoise=denoise,
    )
    meta_path, x_path = perframe_grid_cache_paths(cache_dir, rate_ms, t_win, stride, grid_dim, use_phase, use_empty_baseline, use_cir, denoise)
    np.save(x_path, X)
    np.savez(meta_path, cells=cells, present=present, groups=groups, activity_codes=activity_codes)
    print(f"Cached dataset to {x_path}")
    return X, cells, present, groups, activity_codes


class SlidingWindowCaptureDataset(Dataset):
    """On-the-fly sliding-window Dataset over build_capture_cache()'s
    per-capture arrays -- generates each window by SLICING the small,
    already-cached per-capture feature array at __getitem__ time, instead
    of pre-materializing every overlapping window as its own on-disk copy
    (see build_capture_cache's docstring for why that matters at small
    strides -- stride=1 over this pipeline's ~50 captures would otherwise
    need ~108GB of overlapping, 98%-redundant copies).

    Same per-window contents as make_perframe_windows/CSIDataset with
    extra=present (extra2=condition): (window, pos_seq, present_seq) or
    (window, pos_seq, present_seq, condition) when condition_value is given.
    """

    def __init__(
        self,
        capture_cache: dict[str, dict],
        trial_keys: list[str],
        t_win: int,
        stride: int,
        condition_value: dict[str, float] | None = None,
    ):
        self.capture_cache = capture_cache
        self.t_win = t_win
        self.condition_value = condition_value
        self.index: list[tuple[str, int]] = []
        for tk in trial_keys:
            n = capture_cache[tk]["feat"].shape[-1]
            for start in range(0, n - t_win + 1, stride):
                self.index.append((tk, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        tk, start = self.index[i]
        end = start + self.t_win
        entry = self.capture_cache[tk]
        window = torch.as_tensor(np.array(entry["feat"][:, :, start:end]), dtype=torch.float32)
        pos_seq = torch.as_tensor(np.nan_to_num(entry["pos"][start:end], nan=0.5), dtype=torch.float32)
        present_seq = torch.as_tensor(entry["present"][start:end], dtype=torch.float32)
        if self.condition_value is not None:
            cond = torch.as_tensor(self.condition_value[tk], dtype=torch.float32)
            return window, pos_seq, present_seq, cond
        return window, pos_seq, present_seq
