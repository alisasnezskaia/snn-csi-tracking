"""Interactive floor-plane homography calibration: click known floor points in
a reference frame, pair each with its real-world (X, Y) floor coordinate
(meters, from the room's floor-plan diagram), and solve for the pixel<->floor
homography. This lets cached trajectory .npy files (currently relative pixel
space) get converted into calibrated real-world coordinates -- see
apply_homography_to_trajectories.py.

Pick points that are:
  - actually ON THE FLOOR (not desktop corners -- those sit ~0.75m above the
    floor plane and would bias the homography if treated as floor points)
  - spread out across the room, not clustered in one area (a tight cluster
    makes the homography numerically unstable / only accurate near that spot)
  - things you can confidently assign a real-world (X, Y) to from the
    diagram/room measurements -- if you're not sure of a point's real-world
    position, skip it rather than guess; a bad correspondence is worse than
    one fewer point (need >=4 total)

Suggested reference points to consider (pick whichever you can actually
confirm real-world coordinates for): wall-floor corners, the door threshold,
desk-leg floor contact points (not the desktop), any visible floor tile/mat
seams. Avoid the near-camera equipment desk area (ROI_EXCLUDE in
build_trajectory_labels.py) entirely, and avoid points inside the alcove
(Tx/shield nook) unless you've confirmed that nook is actually visible in
this camera's frame.

Run:
    .venv/bin/python scripts/calibrate_homography.py <path/to/frame_or_video>
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

path = Path(sys.argv[1])
if path.suffix.lower() in (".mp4", ".avi", ".mov"):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)  # a few seconds in, not the first (often blank/transitioning) frame
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"couldn't read a frame from {path}")
else:
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"couldn't read image {path}")

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(figsize=(12, 7))
ax.imshow(rgb)
ax.set_title(
    "Click floor-level reference points, in the SAME order you'll enter\n"
    "their real-world (X,Y) next. Press Enter or close the window when done."
)
print("Click each reference point in the image window, in order (floor-level points only).")
print("Press Enter (or close the window) when you're finished clicking.")
pts = plt.ginput(n=-1, timeout=0)
plt.close(fig)

if len(pts) < 4:
    raise SystemExit(f"Need at least 4 points for a homography, got {len(pts)}. Re-run and click more.")

pixel_pts = np.array(pts, dtype=np.float64)
print(f"\nCollected {len(pixel_pts)} pixel points:")
for i, (x, y) in enumerate(pixel_pts):
    print(f"  point {i}: pixel=({x:.1f}, {y:.1f})")

print("\nNow enter each point's real-world floor coordinate in meters (from the room")
print("diagram), same order as clicked. Example input for a point: 1.8 0.6")
world_pts = []
for i in range(len(pixel_pts)):
    raw = input(f"  point {i} real-world (X Y) meters: ")
    x, y = map(float, raw.split())
    world_pts.append((x, y))
world_pts = np.array(world_pts, dtype=np.float64)

method = cv2.RANSAC if len(pixel_pts) > 4 else 0  # RANSAC only meaningful with a spare point or more
H, mask = cv2.findHomography(pixel_pts, world_pts, method=method)
print("\nHomography matrix (pixel -> floor meters):")
print(H)

# Validation: reproject the same pixel points through H and compare to the
# real-world coords you entered. Large errors mean either a bad point pick,
# a point that wasn't really on the floor plane, or a real-world coordinate
# that doesn't match the diagram as well as you thought.
reproj = cv2.perspectiveTransform(pixel_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
errors = np.linalg.norm(reproj - world_pts, axis=1)
print("\nReprojection error per point (meters) -- should be small (a few cm):")
for i, e in enumerate(errors):
    flag = "  <- check this one" if e > 0.15 else ""
    print(f"  point {i}: {e:.3f} m{flag}")
print(f"mean: {errors.mean():.3f} m, max: {errors.max():.3f} m")

out_path = Path(__file__).parent.parent / "data" / "processed" / "floor_homography.npy"
np.save(out_path, H)
print(f"\nSaved homography to {out_path}")
