"""Build per-CSI-frame (x, y) position traces to support a motion/displacement
regression target (not absolute position -- see conversation notes on why
single-link CSI can't support fine-grained localization, and why "how much/
fast is the person moving" is the more physically-grounded, more feasible
target given the time budget).

For each trial in our existing 50ms training data (data/raw/{NLoS,PLoS}/csi_
office_50ms_interframe.mat), find its matching raw session on the Drive
(NLoS -> data/NoShield/, PLoS -> data/Shield/, confirmed mapping), download
that trial's video + .pcap, extract a pixel trajectory via MediaPipe pose
(hip midpoint, falling back to nose when hips are low-confidence/occluded --
see extract_pose_trajectory), and align it to CSI frame times using:

    csi_frame_time(i)   = pcap_first_packet_time + i * (interframe_ms / 1000)
    video_frame_time(j) = video_start_time (from filename) + j / fps

validated on one session: ~0.5s offset between pcap start and video start,
durations agree to within ~0.2s. That offset is a non-issue here since the
downstream speed label gets aggregated over whole windows (~seconds), not
read frame-by-frame.

Output: one .npy per trial at data/processed/trajectories/<condition>_<rate>
ms_<activity_code>_<capture_filename>.npy, shape (NumCSI, 2) = (x_px, y_px)
position per CSI frame (NaN where nothing was detected), still at native CSI
frame resolution -- converting this into a per-window speed regression target
(smooth, differentiate, aggregate over a window) is a separate, later step
once a window size is chosen, so this cache doesn't need rebuilding if that
choice changes.

Run:
    .venv/bin/python scripts/build_trajectory_labels.py [rate_ms]
Defaults to 50ms.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from scapy.utils import PcapReader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from snn_csi_tracking.data import mat_loader

# Near-camera monitor SCREEN ONLY -- not the desk, not the bottle/papers/
# keyboard around it. A real person does sit at that desk in some trials
# (confirmed against another sample video), so the desk is legitimate
# activity space and must NOT be excluded -- only the screen itself, since
# it shows changing/flickering content and any detection landing exactly on
# its pixels is definitionally wrong (a screen is never a person). See
# scripts/calibrate_roi_exclude.py.
#
# No fallback box here on purpose: an earlier draft used a whole-desk
# exclusion box, which would have deleted real sitting-trial ground truth --
# worse than doing no exclusion at all. If the screen polygon hasn't been
# calibrated yet, we skip the exclusion entirely (screen-flicker false
# positives may slip through uncaught) rather than fall back to something
# that actively destroys good data.
ROI_EXCLUDE_POLY_PATH = Path(__file__).parent.parent / "data" / "processed" / "roi_exclude_polygon.npy"

if ROI_EXCLUDE_POLY_PATH.exists():
    _roi_polygon = np.load(ROI_EXCLUDE_POLY_PATH).astype(np.float32)
else:
    _roi_polygon = None
    print(
        f"WARNING: {ROI_EXCLUDE_POLY_PATH} not found -- run scripts/calibrate_roi_exclude.py "
        f"first (trace the monitor SCREEN only). Proceeding with NO screen exclusion for now."
    )


def _in_excluded_zone(x: float, y: float) -> bool:
    if _roi_polygon is None:
        return False
    return cv2.pointPolygonTest(_roi_polygon, (x, y), False) >= 0

# Rough heuristic to catch obviously-wrong detections (e.g. a false positive
# on the excluded screen bleeding just outside ROI_EXCLUDE, or a momentary
# mis-track), not a precise physical bound -- tune after visually reviewing
# a few extracted trajectories. Expressed per-frame at 30fps and scaled to
# each video's actual fps below.
MAX_JUMP_PX_AT_30FPS = 250

RATE_MS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
REMOTE = "gdrive csi:"
DATA_ROOT = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed" / "trajectories"

# NLoS/PLoS (our aggregated training data) -> raw Drive folder name, confirmed:
# NLoS = No Shield, PLoS = Partial Shield.
CONDITION_TO_DRIVE = {"NLoS": "NoShield", "PLoS": "Shield"}

ACTIVITY_FOLDER = {
    "E": "E - Empty room",
    "EN-S": "EN-S - Person enters and sits down",
    "EN-W": "EN-W - Person enters and Walks around",
    "L": "L - Person stands up and Leaves",
    "S": "S - Person Sitting",
}


def rclone_lsf(remote_path: str, dirs_only: bool = False) -> list[str]:
    cmd = ["rclone", "lsf", remote_path, "--drive-shared-with-me"]
    if dirs_only:
        cmd.append("--dirs-only")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line]


def find_activity_folder(rate_base: str, activity_code: str) -> str:
    """Match the activity folder by code prefix (e.g. "E - ", "EN-S - "),
    case-insensitively -- Shield and NoShield don't always agree on
    capitalization (confirmed: NoShield has "E - Empty room", Shield has
    "E - Empty Room"), so a hardcoded exact name breaks on one of them.
    """
    prefix = (activity_code + " - ").lower()
    for folder in rclone_lsf(f"{REMOTE}{rate_base}/", dirs_only=True):
        name = folder.rstrip("/")
        if name.lower().startswith(prefix):
            return name
    raise ValueError(f"no activity folder starting with {activity_code!r} found under {rate_base}")


def rclone_copy(remote_path: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rclone", "copy", remote_path, str(local_dir), "--drive-shared-with-me"],
        check=True,
        capture_output=True,
    )


def video_start_time(filename: str) -> datetime:
    # TimeVideo_YYYYMMDD_HHMMSS.mp4
    m = re.match(r"TimeVideo_(\d{8})_(\d{6})\.mp4", filename)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def pcap_first_packet_time(pcap_path: Path) -> float:
    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            return float(pkt.time)
    raise ValueError(f"no packets in {pcap_path}")


_mp_pose = mp.solutions.pose


BASE_MEASUREMENT_NOISE = 4.0
NOSE_MEASUREMENT_NOISE_MULT = 10.0  # hip and nose are different physical points
                                    # on the body (different height, some
                                    # horizontal offset too) -- switching
                                    # sources mid-trial showed up as a visible
                                    # step/jump in the position trace, not real
                                    # motion. Trusting nose-based measurements
                                    # much less lets the filter lean on its own
                                    # motion model through a source switch
                                    # instead of jumping straight to the new
                                    # point.


def _kalman_smooth(positions: np.ndarray, source: np.ndarray | None = None) -> np.ndarray:
    """Constant-velocity Kalman filter over a raw (possibly gappy) (N, 2)
    position sequence. Predicts every frame, corrects only on frames with a
    real detection -- fills gaps using the motion model instead of plain
    linear interpolation, and denoises jitter on valid frames too (unlike
    interpolation, which just trusts both endpoints of a gap outright).
    Leading frames before the first valid detection are left as NaN (same
    convention as before -- align_to_csi/downstream still handles those).

    source: optional (N,) array of 0=hip/1=nose per frame (ignored where NaN).
    Nose-sourced measurements get inflated measurement noise (see
    NOSE_MEASUREMENT_NOISE_MULT) so a hip<->nose fallback switch doesn't read
    to the filter as a sudden real displacement.
    """
    n = len(positions)
    valid = ~np.isnan(positions[:, 0])
    if valid.sum() < 2:
        return positions

    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    base_noise = np.eye(2, dtype=np.float32) * BASE_MEASUREMENT_NOISE

    first_idx = np.flatnonzero(valid)[0]
    kf.statePost = np.array(
        [[positions[first_idx, 0]], [positions[first_idx, 1]], [0], [0]], dtype=np.float32
    )

    smoothed = positions.copy()
    for i in range(first_idx, n):
        pred = kf.predict()
        if valid[i]:
            mult = NOSE_MEASUREMENT_NOISE_MULT if (source is not None and source[i] == 1) else 1.0
            kf.measurementNoiseCov = base_noise * mult
            meas = np.array([[positions[i, 0]], [positions[i, 1]]], dtype=np.float32)
            corrected = kf.correct(meas)
            smoothed[i] = corrected[:2, 0]
        else:
            smoothed[i] = pred[:2, 0]
    return smoothed


def extract_pose_trajectory(video_path: Path) -> tuple[np.ndarray, float]:
    """Returns (positions, fps) where positions is (NumFrames, 2) = (x_px, y_px),
    NaN where nothing usable was detected (before Kalman-filling).

    MediaPipe pose, hip midpoint primary (stable center-of-mass proxy across
    standing/sitting/bending) falling back to the nose landmark when hips are
    low-confidence (e.g. occluded by a desk while seated) -- feet are the
    least reliable (most occluded by furniture/frame edges) so deliberately
    not used at all.

    Two extra robustness passes, both motivated by real failure modes seen on
    a sample trial (results/figures/08_video_trajectory_pose.png -- frequent
    on/off confidence flicker and a few implausible position teleports):
      - _in_excluded_zone rejects any detection landing on the near-camera
        monitor screen itself (not the desk around it -- see the
        ROI_EXCLUDE_POLY_PATH comment above).
      - a max-jump gate rejects frame-to-frame displacements too large to be
        real motion (likely a spurious detection), scaled to this video's fps.
    Both rejections fall back to NaN, same as "nothing detected".
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    LEFT_HIP, RIGHT_HIP = _mp_pose.PoseLandmark.LEFT_HIP, _mp_pose.PoseLandmark.RIGHT_HIP
    NOSE = _mp_pose.PoseLandmark.NOSE
    max_jump = MAX_JUMP_PX_AT_30FPS * (30.0 / max(fps, 1e-6))

    raw = []
    source = []  # 0=hip, 1=nose, nan=no usable detection this frame
    last_valid = None
    with _mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            pos = (np.nan, np.nan)
            src = np.nan
            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                left, right = lm[LEFT_HIP], lm[RIGHT_HIP]
                hip_conf = (left.visibility + right.visibility) / 2
                if hip_conf > 0.5:
                    pos = ((left.x + right.x) / 2 * w, (left.y + right.y) / 2 * h)
                    src = 0
                elif lm[NOSE].visibility > 0.5:
                    pos = (lm[NOSE].x * w, lm[NOSE].y * h)
                    src = 1

            if not np.isnan(pos[0]):
                if _in_excluded_zone(pos[0], pos[1]):
                    pos, src = (np.nan, np.nan), np.nan  # landed on the excluded monitor screen
                elif last_valid is not None and np.hypot(pos[0] - last_valid[0], pos[1] - last_valid[1]) > max_jump:
                    pos, src = (np.nan, np.nan), np.nan  # implausible teleport, likely a false detection

            if not np.isnan(pos[0]):
                last_valid = pos
            raw.append(pos)
            source.append(src)
    cap.release()

    positions = _kalman_smooth(np.array(raw, dtype=np.float64), source=np.array(source, dtype=np.float64))
    return positions, fps


def align_to_csi(
    csi_times: np.ndarray, video_start: datetime, positions: np.ndarray, fps: float
) -> np.ndarray:
    """Nearest-frame match from each CSI frame time to a video frame, with
    linear interpolation across NaN gaps in the pose trajectory first."""
    n_video = len(positions)
    video_times = np.array([(video_start.timestamp() + j / fps) for j in range(n_video)])

    # fill NaN gaps by linear interpolation (leaves leading/trailing NaN as-is)
    x, y = positions[:, 0], positions[:, 1]
    valid = ~np.isnan(x)
    if valid.sum() >= 2:
        x = np.interp(np.arange(n_video), np.flatnonzero(valid), x[valid], left=np.nan, right=np.nan)
        y = np.interp(np.arange(n_video), np.flatnonzero(valid), y[valid], left=np.nan, right=np.nan)

    idx = np.searchsorted(video_times, csi_times).clip(0, n_video - 1)
    return np.stack([x[idx], y[idx]], axis=1)


def process_trial(condition: str, activity_code: str, filename: str, num_csi: int) -> np.ndarray | None:
    drive_condition = CONDITION_TO_DRIVE[condition]
    rate_base = f"data/{drive_condition}/{RATE_MS}ms-interframe/1min"
    activity_folder = find_activity_folder(rate_base, activity_code)
    session_base = f"{rate_base}/{activity_folder}"

    sessions = rclone_lsf(f"{REMOTE}{session_base}/", dirs_only=True)
    if len(sessions) != 1:
        print(f"  WARNING: expected 1 session folder, found {len(sessions)}: {sessions}")
    session = sessions[0].rstrip("/")
    session_path = f"{session_base}/{session}"

    videos = sorted(v for v in rclone_lsf(f"{REMOTE}{session_path}/") if v.endswith(".mp4"))
    capture_idx = int(re.match(r"capture(\d+)\.mat", filename).group(1))
    if capture_idx > len(videos):
        print(f"  WARNING: capture{capture_idx} has no matching video ({len(videos)} videos found)")
        return None
    video_name = videos[capture_idx - 1]  # capture1 <-> earliest video, by convention

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        rclone_copy(f"{REMOTE}{session_path}/{video_name}", tmp_dir)
        rclone_copy(f"{REMOTE}{session_path}/1/capture{capture_idx}.pcap", tmp_dir)

        pcap_t0 = pcap_first_packet_time(tmp_dir / f"capture{capture_idx}.pcap")
        csi_times = pcap_t0 + np.arange(num_csi) * (RATE_MS / 1000)

        positions, fps = extract_pose_trajectory(tmp_dir / video_name)
        v_start = video_start_time(video_name)
        return align_to_csi(csi_times, v_start, positions, fps)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    trials = []
    for condition in ("NLoS", "PLoS"):
        path = DATA_ROOT / condition / f"csi_office_{RATE_MS}ms_interframe.mat"
        if not path.exists():
            continue
        for row in mat_loader.load_table_metadata(path):
            if row["activity_code"] in ACTIVITY_FOLDER:
                trials.append((condition, path, row))

    print(f"{len(trials)} trials to process for {RATE_MS}ms")
    for i, (condition, path, row) in enumerate(trials):
        shape = mat_loader.peek_csi_shape(path, row["csi_key"], row["row_index"])
        out_path = OUT_DIR / f"{condition}_{RATE_MS}ms_{row['activity_code']}_{row['filename']}.npy"
        if out_path.exists():
            print(f"[{i+1}/{len(trials)}] {out_path.name}: cached, skipping")
            continue
        if shape is None:
            print(f"[{i+1}/{len(trials)}] {condition}/{row['filename']}/{row['activity_code']}: "
                  f"SKIPPED (placeholder row)")
            continue
        num_csi = shape[0]

        t0 = time.time()
        print(f"[{i+1}/{len(trials)}] {condition}/{row['filename']}/{row['activity_code']} "
              f"(num_csi={num_csi})...")
        try:
            traj = process_trial(condition, row["activity_code"], row["filename"], num_csi)
        except Exception as e:
            # one trial failing (network hiccup, an unexpected folder-naming
            # difference like Shield/NoShield's Empty Room case-mismatch,
            # etc.) shouldn't silently kill the whole overnight batch -- log
            # and move on, this trial can be retried on a later run.
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        if traj is None:
            continue
        np.save(out_path, traj)
        detected = (~np.isnan(traj[:, 0])).mean()
        print(f"  -> {out_path.name}: {detected:.1%} frames with a position "
              f"({time.time()-t0:.1f}s, {time.time()-t_start:.1f}s total)")


if __name__ == "__main__":
    main()
