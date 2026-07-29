"""Apply the calibrated floor homography (see calibrate_homography.py) to
every cached trajectory .npy, converting pixel-space (x_px, y_px) into
real-world floor coordinates (x_m, y_m).

Writes alongside the originals with a _meters suffix rather than overwriting
them -- downstream code (preprocessing.speed_labels/position_labels) can be
pointed at the calibrated files once they've been spot-checked, without
losing the original pixel-space trajectories.

Run:
    .venv/bin/python scripts/apply_homography_to_trajectories.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

TRAJ_DIR = Path(__file__).parent.parent / "data" / "processed" / "trajectories"
H_PATH = Path(__file__).parent.parent / "data" / "processed" / "floor_homography.npy"

if not H_PATH.exists():
    raise SystemExit(f"{H_PATH} not found -- run calibrate_homography.py first")

H = np.load(H_PATH)

files = sorted(f for f in TRAJ_DIR.glob("*.npy") if not f.stem.endswith("_meters"))
print(f"{len(files)} trajectory files to convert")

for i, f in enumerate(files):
    positions = np.load(f)  # (N, 2) pixel, NaN where undetected
    valid = ~np.isnan(positions[:, 0])
    out = np.full_like(positions, np.nan)
    if valid.any():
        pts = positions[valid].reshape(-1, 1, 2).astype(np.float64)
        world = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        out[valid] = world
    out_path = f.with_name(f.stem + "_meters.npy")
    np.save(out_path, out)
    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{len(files)}")

print("done")
