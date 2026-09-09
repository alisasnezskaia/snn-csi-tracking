"""Metric depth at the tracked head position, via Depth-Anything-V2's
metric-indoor checkpoint -- gives real camera-relative distance in meters,
which camera_geometry.deproject_pixel_to_camera_frame then combines with
the (u, v) pixel position to recover real-world (X, Y, Z).

Kept in its own module (not trajectory_extraction.py) so the heavy
`transformers` import + model download only happens for callers that
actually enable the 3D extension -- the 2D presence/position pipeline never
imports this file.

Depth-Anything-V2's METRIC checkpoints (unlike the base relative model) are
fine-tuned on NYU-Depth-v2 to output real depth in meters, using the
standard monocular-depth-benchmark convention: depth = Z, the distance
along the camera's optical axis (not radial/Euclidean range from the
camera). Its accuracy is bounded by how well NYU-Depth-v2's indoor scenes
match this room -- there is no per-scene recalibration here, and no
guarantee of frame-to-frame scale consistency beyond what the model itself
provides (it's a single-image model, not a video-consistent one).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEPTH_PATCH_RADIUS_PX = 3  # median over a (2r+1)x(2r+1) patch, not a single noisy pixel


def setup_depth_pipeline():
    import torch
    from transformers import pipeline

    return pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL_NAME,
        device=0 if torch.cuda.is_available() else -1,
    )


def get_depth_at_point(depth_pipe, frame_rgb: np.ndarray, x_norm: float, y_norm: float) -> float:
    """Returns metric depth (meters) at (x_norm, y_norm), as the median over
    a small patch around the point -- a single pixel is noisy, especially
    near a moving person's silhouette edge where depth can jump between the
    person and the background within a few pixels.

    Uses `predicted_depth` (the model's raw per-pixel tensor, already
    resized to the input image's resolution) rather than the pipeline's
    `depth` key, which is a min/max-normalized 0-255 PIL image meant for
    display -- reading depth off that would silently discard the metric
    scale entirely, for both the relative and metric checkpoints.
    """
    from PIL import Image

    output = depth_pipe(Image.fromarray(frame_rgb))
    depth_map = output["predicted_depth"].squeeze().cpu().numpy()
    h, w = depth_map.shape
    px = int(np.clip(x_norm, 0, 1) * (w - 1))
    py = int(np.clip(y_norm, 0, 1) * (h - 1))
    r = DEPTH_PATCH_RADIUS_PX
    patch = depth_map[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1]
    return float(np.median(patch))


def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    """(width, height) in pixels -- needed to turn MediaPipe's normalized
    [0,1] (x, y) back into real pixel coordinates for deprojection."""
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width == 0 or height == 0:
        raise FileNotFoundError(
            f"couldn't read video dimensions from {video_path} (missing or unreadable) -- "
            "deprojection needs the raw video, not just the cached trajectory/depth."
        )
    return width, height


def add_depth_to_trajectory(depth_pipe, video_path: Path, trajectory: np.ndarray) -> np.ndarray:
    """trajectory: (N, 2) normalized xy, at the video's own (frame_skip-decimated)
    resolution -- i.e. the same array trajectory_extraction caches, before
    resample_to_n() stretches it to the CSI frame count. Returns (N,) metric
    depth in meters, NaN wherever the trajectory itself is NaN or no video
    frame maps there. Only computes depth once per unique trajectory index
    (many video frames can map to the same trajectory sample), since depth
    inference is the expensive step here.
    """
    cap = cv2.VideoCapture(str(video_path))
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    depths = np.full(len(trajectory), np.nan)
    if n_video_frames == 0:
        cap.release()
        return depths

    traj_idx_per_frame = np.linspace(0, len(trajectory) - 1, n_video_frames).astype(int)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_idx = traj_idx_per_frame[frame_idx]
        if not np.isnan(trajectory[t_idx, 0]) and np.isnan(depths[t_idx]):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depths[t_idx] = get_depth_at_point(depth_pipe, rgb, *trajectory[t_idx])
        frame_idx += 1
    cap.release()
    return depths


def hampel_filter_depth(depth_raw: np.ndarray, max_segment_len: int = 60, jump_mad_multiplier: float = 6.0) -> np.ndarray:
    """Removes short depth 'islands' bracketed by abrupt jumps on both
    sides, replacing them with NaN and linearly interpolating over the gap.

    Motivated by an observed failure mode: during a fast pose transition,
    MediaPipe's tracked landmark can briefly drift off the
    person's body onto a background wall for ~10-30 frames before snapping
    back -- (x, y) stays smooth throughout (no xy jump to catch), but depth
    cliffs to the wall's unrelated depth, plateaus, then cliffs back.

    A plain rolling-window (classic Hampel) level filter was tried first
    and failed here: this dataset's depth signal is naturally volatile
    (real human depth changes aren't small), so a wide-enough window to
    outnumber a ~30-frame anomaly also pulls in enough *other* real
    variability that the local median stops being a clean baseline.
    Frame-to-frame jump size turned out to separate cleanly instead (this
    trial: typical jumps ~2, but 3-5 isolated jumps up to 148) -- a real
    depth change from someone moving is gradual (bounded by walking speed
    per frame), while a wrong-pixel jump can be almost arbitrary in size.
    Segments longer than max_segment_len are left alone even if
    jump-bracketed, since a longer sustained level shift more likely means
    the person genuinely settled into a new position than a brief glitch.
    """
    depth = depth_raw.copy().astype(float)
    n = len(depth)
    valid_idx = np.where(~np.isnan(depth))[0]
    if len(valid_idx) < 5:
        return depth

    diffs, diff_positions = [], []
    for i in range(1, len(valid_idx)):
        if valid_idx[i] - valid_idx[i - 1] == 1:  # only directly-consecutive samples, not across a gap
            diffs.append(depth[valid_idx[i]] - depth[valid_idx[i - 1]])
            diff_positions.append(valid_idx[i])
    if len(diffs) < 5:
        return depth

    abs_diffs = np.abs(np.array(diffs))
    median = np.median(abs_diffs)
    mad = np.median(np.abs(abs_diffs - median)) + 1e-8
    threshold = median + jump_mad_multiplier * mad * 1.4826  # 1.4826 scales MAD to a std-comparable unit

    cliffs = [pos for pos, d in zip(diff_positions, abs_diffs) if d > threshold]
    if not cliffs:
        return depth

    boundaries = [valid_idx[0], *cliffs, valid_idx[-1] + 1]
    for i in range(1, len(boundaries) - 1):
        seg_start, seg_end = boundaries[i], boundaries[i + 1]
        if seg_end - seg_start <= max_segment_len:
            depth[seg_start:seg_end] = np.nan

    valid = ~np.isnan(depth)
    if valid.sum() >= 2:
        idx = np.arange(n)
        depth[~valid] = np.interp(idx[~valid], idx[valid], depth[valid])
    return depth


def depth_cache_path(cache_dir: Path, trial_key: str) -> Path:
    return cache_dir / f"{trial_key}.npy"


def load_or_extract_depth(video_path: Path, trial_key: str, trajectory: np.ndarray, cache_dir: Path, depth_pipe) -> np.ndarray | None:
    """Returns None when neither a cached depth trace nor the raw video is available."""
    cache_path = depth_cache_path(cache_dir, trial_key)
    if cache_path.exists():
        return np.load(cache_path)

    if not video_path.exists():
        return None

    depths = add_depth_to_trajectory(depth_pipe, video_path, trajectory)
    depths = hampel_filter_depth(depths)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, depths)
    return depths
