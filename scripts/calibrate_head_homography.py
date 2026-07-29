"""Head-height (not floor-plane) homography calibration for the presence+
position pipeline's ground truth -- see
src/snn_csi_tracking/data/trajectory_extraction.py, which tracks the head/
shoulder midpoint (HEAD_LANDMARK_IDS = ears+shoulders) specifically because
it stays visible when the floor or feet don't. That means the usual
floor-plane homography (_removed_legacy/scripts/calibrate_homography.py --
explicitly calibrated against floor-level points, NOT desk-height ones,
"would bias the homography if treated as floor points") is the wrong plane
for our tracked point, which sits ~1.5-1.8m above the floor: applying a
floor homography to a head pixel would reintroduce exactly the bias that
script warns against, worse.

This script instead calibrates directly against head-height observations:
for each of N (>=4, ideally >=6) calibration stops, a person stands still
at a real-world floor position (X, Y) in meters (easy to tape-measure) --
we observe where their HEAD lands on camera, not their feet, so the fitted
homography maps head pixel -> the floor (X, Y) the person was standing on,
skipping any need for floor/feet visibility.

Reuses the exact same landmark set and per-frame extraction convention as
trajectory_extraction.py (HEAD_LANDMARK_IDS, HEAD_Y_NUDGE, normalized
[0,1] coordinates) so the calibration is consistent with how the real
dataset's ground truth was extracted -- without modifying that module.

Calibration points are supplied as (frame_index, world_x_m, world_y_m)
rows in a CSV (no header) against a single calibration video recorded from
the SAME fixed camera position/framing as the real sessions, each frame
picked while the subject stands still at that real-world spot. If
auto-detection fails on a given frame (or --manual is passed), falls back
to an interactive click, same UX as the legacy script.

Run:
    .venv/bin/python scripts/calibrate_head_homography.py \\
        path/to/calibration_video.mp4 path/to/points.csv [--manual]

points.csv format (no header), one row per calibration stop:
    frame_index,world_x_m,world_y_m
    120,0.0,0.0
    340,1.8,0.6
    ...

Output:
    data/processed/head_homography.npy  (3x3 homography, head-pixel[0,1] -> floor meters)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_csi_tracking.data.trajectory_extraction import HEAD_LANDMARK_IDS, HEAD_Y_NUDGE, setup_pose_landmarker

OUT_PATH = REPO_ROOT / "data" / "processed" / "head_homography.npy"


def read_points_csv(path: Path) -> list[tuple[int, float, float]]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            frame_idx, wx, wy = row
            rows.append((int(frame_idx), float(wx), float(wy)))
    return rows


def grab_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"couldn't read frame {frame_idx}")
    return frame


def detect_head_point(frame: np.ndarray, landmarker, mp, timestamp_ms: int) -> tuple[float, float] | None:
    """Single-frame version of trajectory_extraction.extract_head_trajectory's
    per-frame detection -- same landmark set/convention, no video loop."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks[0]
    xs = [lm[i].x for i in HEAD_LANDMARK_IDS]
    ys = [lm[i].y for i in HEAD_LANDMARK_IDS]
    return (float(np.mean(xs)), min(ys) - HEAD_Y_NUDGE)


def manual_click_point(frame: np.ndarray, frame_idx: int) -> tuple[float, float]:
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(rgb)
    ax.set_title(f"frame {frame_idx}: click the head/shoulder point, then close the window")
    pts = plt.ginput(n=1, timeout=0)
    plt.close(fig)
    if not pts:
        raise SystemExit(f"no click registered for frame {frame_idx}")
    px, py = pts[0]
    return (px / w, py / h)  # normalize to the same [0,1] convention as extract_head_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("points_csv", type=Path)
    parser.add_argument("--manual", action="store_true", help="skip auto-detection, click every point by hand")
    args = parser.parse_args()

    calibration_points = read_points_csv(args.points_csv)
    if len(calibration_points) < 4:
        raise SystemExit(f"need >=4 calibration points, got {len(calibration_points)}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"couldn't open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    landmarker, mp = (None, None) if args.manual else setup_pose_landmarker()

    head_pts, world_pts = [], []
    for frame_idx, wx, wy in calibration_points:
        frame = grab_frame(cap, frame_idx)
        point = None
        if not args.manual:
            timestamp_ms = int(frame_idx / fps * 1000)
            point = detect_head_point(frame, landmarker, mp, timestamp_ms)
            if point is None:
                print(f"frame {frame_idx}: auto-detection failed, falling back to manual click")
        if point is None:
            point = manual_click_point(frame, frame_idx)
        print(f"frame {frame_idx}: head=({point[0]:.4f}, {point[1]:.4f})  world=({wx:.3f}, {wy:.3f})m")
        head_pts.append(point)
        world_pts.append((wx, wy))
    cap.release()

    head_pts = np.array(head_pts, dtype=np.float64)
    world_pts = np.array(world_pts, dtype=np.float64)

    method = cv2.RANSAC if len(head_pts) > 4 else 0
    H, _mask = cv2.findHomography(head_pts, world_pts, method=method)
    print("\nHead-height homography (head pixel [0,1] -> floor meters):")
    print(H)

    reproj = cv2.perspectiveTransform(head_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors_m = np.linalg.norm(reproj - world_pts, axis=1)
    print("\nReprojection error per point (this is the calibration's own precision floor):")
    for i, e in enumerate(errors_m):
        flag = "  <- check this one" if e > 0.15 else ""
        print(f"  point {i}: {e * 100:.1f} cm{flag}")
    print(f"mean: {errors_m.mean() * 100:.1f} cm, max: {errors_m.max() * 100:.1f} cm")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, H)
    print(f"\nSaved homography to {OUT_PATH}")


if __name__ == "__main__":
    main()
