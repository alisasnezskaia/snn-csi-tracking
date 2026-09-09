"""Head-position trajectory extraction from the raw-capture videos, via
MediaPipe's Tasks API pose landmarker (not the legacy `mp.solutions` API).

Tracks shoulder/ear landmarks (visible from the back too, unlike a face
detector) as a head-position proxy, then interpolates short detector-noise
gaps while preserving long ones (person genuinely not in frame yet /
already left -- real ground truth, not to be smoothed away).

Separate from data/processed/trajectories/ (the grid-motion pipeline's own,
differently-extracted trajectory cache, built by
scripts/build_trajectory_labels.py) -- this module writes to
data/processed/presence_position_trajectories/ instead, to avoid
cross-contaminating the two pipelines' caches.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_LANDMARKER_PATH = Path(__file__).parent.parent.parent.parent / "models" / "pose_landmarker_lite.task"

HEAD_LANDMARK_IDS = (7, 8, 11, 12)  # ears + shoulders -- visible from front OR back
HEAD_Y_NUDGE = 0.05  # shoulder-midpoint y, nudged upward toward the head


def setup_pose_landmarker():
    POSE_LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not POSE_LANDMARKER_PATH.exists():
        urllib.request.urlretrieve(POSE_LANDMARKER_URL, POSE_LANDMARKER_PATH)

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base_options = mp_python.BaseOptions(model_asset_path=str(POSE_LANDMARKER_PATH))
    options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
    landmarker = vision.PoseLandmarker.create_from_options(options)
    return landmarker, mp


def extract_head_trajectory(video_path: str, landmarker, mp, frame_skip: int = 2) -> tuple[np.ndarray, float, int]:
    """Returns (positions (N,2) normalized xy with NaN where undetected, fps, n_frames_total)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    positions: list = [None] * n_frames_total

    frame_idx, timestamp_ms = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                xs = [lm[i].x for i in HEAD_LANDMARK_IDS]
                ys = [lm[i].y for i in HEAD_LANDMARK_IDS]
                positions[frame_idx] = (np.mean(xs), min(ys) - HEAD_Y_NUDGE)
        timestamp_ms += int(1000 / fps) * frame_skip
        frame_idx += 1

    cap.release()
    positions_arr = np.array([p if p is not None else (np.nan, np.nan) for p in positions])
    return positions_arr, fps, n_frames_total


def clean_trajectory(positions: np.ndarray, max_interp_gap: int = 30) -> tuple[np.ndarray, int]:
    """Interpolates short gaps (detector noise, <~1s) but preserves long gaps
    (person genuinely not in frame yet / already left)."""
    df = pd.DataFrame(positions, columns=["x", "y"])
    nan_mask = df["x"].isna()

    runs, start = [], None
    for i, is_nan in enumerate(nan_mask):
        if is_nan and start is None:
            start = i
        elif not is_nan and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(nan_mask) - start))

    df_interp = df.interpolate(limit=max_interp_gap, limit_area="inside")
    for gap_start, gap_len in runs:
        if gap_len > max_interp_gap:
            df_interp.iloc[gap_start : gap_start + gap_len] = np.nan

    df_smooth = df_interp.rolling(window=5, center=True, min_periods=1).mean()
    n_missing = int(df_interp["x"].isna().sum())
    return df_smooth[["x", "y"]].to_numpy(), n_missing


def classify_presence_from_gaps(positions: np.ndarray) -> np.ndarray:
    """Ground-truth presence, smarter than the naive `~isnan(positions[:,0])`:
    that naive rule treats EVERY undetected frame as "person absent," which
    is correct for a gap touching the very start or
    end of a recording (genuinely not yet entered / already left) but wrong
    for a gap stranded in the MIDDLE of a recording (detected before it AND
    after it) -- during an activity where the person never left, a middle
    gap means the whole-body pose detector lost tracking for a while (e.g.
    mid-stride, turned away, crouched), not that the person vanished. The
    head/ear/shoulder landmarks used here (extract_head_trajectory) can be
    confirmed still visible in the source video even when the whole-body
    pose skeleton the detector needs first fails to lock on.

    Returns a bool array, True = present. Position stays NaN/unknown for
    reclassified middle gaps (only the presence label changes, not a
    fabricated position) -- callers should still mask position loss/metrics
    by their own present array, same convention as before.
    """
    nan_mask = np.isnan(positions[:, 0])
    n = len(nan_mask)
    present = ~nan_mask.copy()

    i = 0
    while i < n:
        if nan_mask[i]:
            j = i
            while j < n and nan_mask[j]:
                j += 1
            touches_start, touches_end = (i == 0), (j == n)
            if not touches_start and not touches_end:
                present[i:j] = True  # interior gap -- detected before and after, likely still present
            i = j
        else:
            i += 1
    return present


def trajectory_cache_path(cache_dir: Path, trial_key: str) -> Path:
    return cache_dir / f"{trial_key}.npy"


def load_or_extract_trajectory(video_path: Path, trial_key: str, cache_dir: Path, landmarker, mp, frame_skip: int = 2) -> np.ndarray | None:
    """Returns None (rather than raising) when neither a cached trajectory
    nor the raw video is available -- most trials only carry the cache
    forward once extracted, see discover_captures()."""
    cache_path = trajectory_cache_path(cache_dir, trial_key)
    if cache_path.exists():
        return np.load(cache_path)

    if not video_path.exists():
        return None

    positions_raw, _fps, _n_frames = extract_head_trajectory(str(video_path), landmarker, mp, frame_skip=frame_skip)
    positions_clean, _n_missing = clean_trajectory(positions_raw)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, positions_clean)
    return positions_clean
