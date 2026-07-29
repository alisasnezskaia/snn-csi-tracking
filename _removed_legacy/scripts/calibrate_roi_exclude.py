"""Interactive ROI-exclusion polygon calibration: click points tracing the
outline of the near-camera monitor SCREEN ONLY (the physical display panel --
not the desk, not the bottle/papers/keyboard around it) in a reference frame,
and save the polygon for use by extract_pose_trajectory's exclusion check.

IMPORTANT -- screen only, not the desk: a real person does sit at that desk
in some trials (confirmed against another sample video), so the desk itself
is legitimate activity space and must NOT be excluded. The screen is
different -- it shows changing/flickering content, so any pose detection
landing exactly on its pixels is definitionally wrong (a screen is never a
person), regardless of what else is happening at the desk around it. Keep
the traced polygon tight to just the visible screen/bezel boundary.

Click points in order tracing the screen's outline, then press Enter or
close the window.

Run:
    .venv/bin/python scripts/calibrate_roi_exclude.py <path/to/frame_or_video>
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
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
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
    "Click points tracing the MONITOR SCREEN's outline ONLY (not the desk/bottle/papers),\n"
    "in order around its boundary. Press Enter or close the window when done."
)
print("Click points around the monitor SCREEN's outline only (not the desk), in order.")
print("Press Enter (or close the window) when finished.")
pts = plt.ginput(n=-1, timeout=0)
plt.close(fig)

if len(pts) < 3:
    raise SystemExit(f"Need at least 3 points for a polygon, got {len(pts)}.")

polygon = np.array(pts, dtype=np.float64)
print(f"\nCollected {len(polygon)} polygon points:")
for i, (x, y) in enumerate(polygon):
    print(f"  point {i}: ({x:.1f}, {y:.1f})")

# quick visual confirmation
fig, ax = plt.subplots(figsize=(12, 7))
ax.imshow(rgb)
closed = np.vstack([polygon, polygon[0]])
ax.plot(closed[:, 0], closed[:, 1], "r-", linewidth=2)
ax.fill(polygon[:, 0], polygon[:, 1], color="red", alpha=0.25)
ax.set_title("Confirm: red area is what gets excluded")
out_preview = Path(__file__).parent.parent / "results" / "figures" / "11_roi_polygon_preview.png"
fig.savefig(out_preview, dpi=130)
plt.close(fig)
print(f"\nSaved preview to {out_preview} -- check it looks right before trusting this.")

out_path = Path(__file__).parent.parent / "data" / "processed" / "roi_exclude_polygon.npy"
np.save(out_path, polygon)
print(f"Saved polygon to {out_path}")
