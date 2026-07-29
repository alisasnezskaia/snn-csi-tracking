"""Loading of the raw, plain-text CSI capture files (not MCOS `.mat` tables).

Layout expected under `data/raw_captures/` (populated manually, one file per
condition x activity x repeat -- see mat_loader.py's docstring for the
different, MCOS-table-based `data/raw/` layout used by the grid-motion
pipeline; the two are unrelated data sources from the same capture campaign):

    data/raw_captures/
        {condition}_{activity}/
            capture1.mat ... capture5.mat
            videos/
                segment1.mp4 ... segment5.mp4

Despite the ".mat" extension, each capture file is plain text: one line per
(frame, antenna) reading. `tokens[:27]` is metadata (frame id, antenna,
subcarrier count, timestamp, ...), `tokens[27:]` is 1024 complex subcarrier
values as interleaved (real, imag) pairs -- confirmed via a guard-band check
against the known 1024-subcarrier count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from snn_csi_tracking.data.preprocessing import denoise_amplitude

CSI_HEADER_LEN = 27
CSI_TIMESTAMP_IDX = 12
CSI_EXPECTED_TOKENS = 2075

CONDITIONS = ("NLoS", "PLoS")
ACTIVITY_CODES = ("E", "EN-S", "EN-W", "L", "S")
NUM_ANTENNAS = 3
NUM_SUBCARRIERS = 1024


@dataclass
class Capture:
    condition: str
    activity: str
    capture_name: str  # e.g. "capture1"
    csi_path: Path
    video_path: Path

    @property
    def trial_key(self) -> str:
        return f"{self.condition}_{self.activity}_{self.capture_name}"


def discover_captures(root: Path) -> list[Capture]:
    """Walks `root` (expected: data/raw_captures/) for every
    {condition}_{activity}/capture{n}.mat. `video_path` is filled in
    regardless of whether the file actually exists -- most trials only have
    a cached trajectory (data/processed/presence_position_trajectories/),
    not the raw video, since videos aren't kept in Drive long-term once
    their trajectory has been extracted. Callers needing the video (to
    extract a trajectory not already cached) must check existence
    themselves."""
    captures = []
    for condition in CONDITIONS:
        for activity in ACTIVITY_CODES:
            group_dir = root / f"{condition}_{activity}"
            if not group_dir.is_dir():
                continue
            for csi_path in sorted(group_dir.glob("capture*.mat")):
                capture_name = csi_path.stem
                idx = capture_name.replace("capture", "")
                video_path = group_dir / "videos" / f"segment{idx}.mp4"
                captures.append(Capture(condition, activity, capture_name, csi_path, video_path))
    return captures


def load_empty_room_baseline(
    root: Path, condition: str, denoise: str | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-(antenna, subcarrier) amplitude mean/std across every frame of
    every genuine empty-room ("E") capture for this condition -- a
    person-free reference profile, uncontaminated by any trial's own
    occupancy, to normalize against instead of a trial's own (possibly-
    occupied) mean/std across time (see presence_position_dataset.
    compute_features's `baseline` arg).

    denoise: None, "wavelet", or "pca" (preprocessing.denoise_amplitude) --
    must match whatever compute_features denoises the live signal with,
    since denoising only one side of the subtraction would reintroduce
    exactly the mismatch the baseline is meant to remove.

    Returns (mean, std), each (NUM_ANTENNAS, NUM_SUBCARRIERS, 1) for
    broadcasting against a (Ant, Sub, T) csi array, or None if this
    condition has no "_E" captures.
    """
    group_dir = root / f"{condition}_E"
    if not group_dir.is_dir():
        return None
    amps = []
    for csi_path in sorted(group_dir.glob("capture*.mat")):
        csi, _timestamps = parse_csi_file(csi_path)
        if csi.shape[2] == 0:
            continue
        amps.append(denoise_amplitude(np.abs(csi), denoise, freq_axis=1, time_axis=2))
    if not amps:
        return None
    all_amp = np.concatenate(amps, axis=2).astype(np.float32)
    mean = all_amp.mean(axis=2, keepdims=True)
    std = all_amp.std(axis=2, keepdims=True)
    return mean, std


def load_empty_room_cir_baseline(root: Path, condition: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Like load_empty_room_baseline, but for channel impulse response (CIR)
    magnitude -- IFFT across the subcarrier axis, per antenna -- instead of
    raw amplitude (see presence_position_dataset.compute_features's
    `use_cir` arg). Returns (mean, std), each (NUM_ANTENNAS, NUM_
    SUBCARRIERS, 1) -- the CIR delay-tap axis has the same length as the
    subcarrier axis it's the IFFT of -- or None if this condition has no
    "_E" captures.
    """
    group_dir = root / f"{condition}_E"
    if not group_dir.is_dir():
        return None
    mags = []
    for csi_path in sorted(group_dir.glob("capture*.mat")):
        csi, _timestamps = parse_csi_file(csi_path)
        if csi.shape[2] == 0:
            continue
        cir = np.fft.ifft(csi, axis=1)
        mags.append(np.abs(cir))
    if not mags:
        return None
    all_mag = np.concatenate(mags, axis=2).astype(np.float32)
    mean = all_mag.mean(axis=2, keepdims=True)
    std = all_mag.std(axis=2, keepdims=True)
    return mean, std


def parse_csi_file(path: Path, expected_tokens: int = CSI_EXPECTED_TOKENS) -> tuple[np.ndarray, np.ndarray]:
    """Parses one raw capture file into a (NUM_ANTENNAS, NUM_SUBCARRIERS, NumFrames)
    complex128 array. Skips malformed/truncated lines rather than crashing --
    a small fraction of lines in some files are truncated by a capture-time
    write cutoff.

    Returns (csi, timestamps): timestamps is (NUM_ANTENNAS, NumFrames), the
    real Unix timestamp for each (antenna, frame) reading.
    """
    with open(path) as f:
        raw_lines = f.read().strip().split("\n")
    lines = [line for line in raw_lines if len(line.split()) == expected_tokens]
    n_frames = len(lines) // NUM_ANTENNAS
    csi = np.zeros((NUM_ANTENNAS, NUM_SUBCARRIERS, n_frames), dtype=np.complex128)
    timestamps = np.zeros((NUM_ANTENNAS, n_frames))
    frame_counter: dict[str, int] = {}
    for line in lines:
        tokens = line.split()
        frame_id, antenna = tokens[0], int(tokens[1])
        ts = float(tokens[CSI_TIMESTAMP_IDX])
        raw = np.array(tokens[CSI_HEADER_LEN:], dtype=np.float64)
        complex_vals = raw[0::2] + 1j * raw[1::2]
        idx = frame_counter.setdefault(frame_id, len(frame_counter))
        if idx < n_frames:
            csi[antenna, :, idx] = complex_vals
            timestamps[antenna, idx] = ts
    return csi, timestamps
