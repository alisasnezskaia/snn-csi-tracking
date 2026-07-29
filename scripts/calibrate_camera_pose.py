"""Full camera-pose calibration from floor-plane correspondences -- the
upgrade over a flat single-plane homography (calibrate_head_homography.py)
made possible once real room geometry (the office floor plan) became
available: instead of needing a dedicated head-height calibration
recording, we use floor-level landmarks already visible in EXISTING
session footage (desk legs, the alcove partition corner, wall corners),
matched to their real-world (X, Y) from the floor plan, and recover the
camera's full pose via cv2.solvePnP (see
src/snn_csi_tracking/data/camera_geometry.py).

With the camera's pose known, ANY pixel can be projected to a real-world
(X, Y) at any assumed height via ray-plane intersection -- not just the
floor -- which is what lets this handle the head/shoulder ground-truth
landmark (well above the floor) without a separate calibration walk.

You click each floor-level point on the reference frame; real-world
coordinates are pre-filled from the floor-plan reading (edit the list
below to match your own corrected readout before running) so you don't
have to retype them.

Run:
    .venv/bin/python scripts/calibrate_camera_pose.py path/to/reference_frame.png

Output:
    data/processed/camera_pose.npz  (K, rvec, tvec, image_width, image_height)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_csi_tracking.data.camera_geometry import (
    camera_center_world, estimate_camera_matrix, reprojection_error_m, solve_camera_pose,
)

OUT_PATH = REPO_ROOT / "data" / "processed" / "camera_pose.npz"

# Real-world floor-plane (X, Y) in meters, read off the floor plan -- EDIT THIS to match
# whatever corrections you made to the coordinate readout, in the same order you'll click
# the matching points on the reference frame. Only include points you can confidently
# click AND confidently place on the floor plan (skip anything uncertain -- a bad
# correspondence is worse than one fewer point, same rule as the legacy calibration
# script). Floor-level corners of the two desks are the best bet -- clearly visible,
# floor-level, and were medium/low confidence only because of the SKETCH reading, not
# because the desks themselves are hard to place.
WORLD_POINTS_M = [
    # (label, world_x, world_y)
    ("desk1_front_left_leg", 2.2, 1.6),
    ("desk1_front_right_leg", 3.9, 1.6),
    ("desk2_front_left_leg", 4.1, 1.7),
    ("desk2_front_right_leg", 5.9, 1.7),
    ("alcove_corner", 1.8, 1.8),
]


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: calibrate_camera_pose.py path/to/reference_frame.png (or .mp4)")
    path = Path(sys.argv[1])

    if path.suffix.lower() in (".mp4", ".avi", ".mov"):
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"couldn't read a frame from {path}")
    else:
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"couldn't read image {path}")

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print(f"Reference frame: {w}x{h}")
    print(f"Click these {len(WORLD_POINTS_M)} points IN ORDER, at floor level:")
    for label, wx, wy in WORLD_POINTS_M:
        print(f"  - {label}  (world = {wx:.2f}, {wy:.2f} m)")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(rgb)
    ax.set_title(
        "Click each point listed in the terminal, IN ORDER, at floor level "
        "(where the object touches the ground, not its tabletop/other height).\n"
        "Press Enter or close the window when done."
    )
    pts = plt.ginput(n=len(WORLD_POINTS_M), timeout=0)
    plt.close(fig)

    if len(pts) < 4:
        raise SystemExit(f"need >=4 points for solvePnP, got {len(pts)}")

    image_points = np.array(pts, dtype=np.float64)
    world_points = np.array([[wx, wy] for _label, wx, wy in WORLD_POINTS_M[: len(pts)]], dtype=np.float64)

    K = estimate_camera_matrix(w, h)
    rvec, tvec = solve_camera_pose(image_points, world_points, K)

    cam_pos = camera_center_world(rvec, tvec)
    print(f"\nRecovered camera position (world X, Y, height): "
          f"({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}) m")
    print("Sanity check: does (X, Y) roughly match where you placed 'C' on the floor plan?")

    errors = reprojection_error_m(image_points, world_points, K, rvec, tvec)
    print("\nFloor reprojection error per point (this calibration's own precision floor):")
    for (label, _wx, _wy), e in zip(WORLD_POINTS_M, errors):
        flag = "  <- check this one" if e > 0.15 else ""
        print(f"  {label}: {e * 100:.1f} cm{flag}")
    print(f"mean: {errors.mean() * 100:.1f} cm, max: {errors.max() * 100:.1f} cm")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, K=K, rvec=rvec, tvec=tvec, image_width=w, image_height=h)
    print(f"\nSaved camera pose to {OUT_PATH}")


if __name__ == "__main__":
    main()
