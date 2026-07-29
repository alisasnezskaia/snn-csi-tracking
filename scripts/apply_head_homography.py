"""Apply the head-height homography (see calibrate_head_homography.py) to
every cached head/shoulder trajectory .npy, converting normalized [0,1]
image-plane (x, y) into real-world floor coordinates (x_m, y_m).

Targets data/processed/presence_position_trajectories/ specifically -- the
head/shoulder cache trajectory_extraction.py builds and the actual
presence+position model trains on -- NOT data/processed/trajectories/ (the
unrelated grid-motion/speed-regression pipeline's own cache, which is what
the legacy _removed_legacy/scripts/apply_homography_to_trajectories.py
operates on).

Writes alongside the originals with a _meters suffix rather than
overwriting them, so the model can be pointed at calibrated labels once
spot-checked, without losing the original normalized trajectories (same
non-destructive convention as the legacy script).

Run:
    .venv/bin/python scripts/apply_head_homography.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
TRAJ_DIR = REPO_ROOT / "data" / "processed" / "presence_position_trajectories"
H_PATH = REPO_ROOT / "data" / "processed" / "head_homography.npy"

if not H_PATH.exists():
    raise SystemExit(f"{H_PATH} not found -- run scripts/calibrate_head_homography.py first")

H = np.load(H_PATH)

files = sorted(f for f in TRAJ_DIR.glob("*.npy") if not f.stem.endswith("_meters"))
print(f"{len(files)} trajectory files to convert")

for i, f in enumerate(files):
    positions = np.load(f)  # (N, 2) normalized [0,1], NaN where undetected
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
