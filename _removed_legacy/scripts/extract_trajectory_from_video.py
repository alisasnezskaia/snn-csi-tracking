"""Prototype: extract a rough (x, y) pixel trajectory from a static-camera video.

Camera is fixed and background is static, so we don't need a pose/detection
model -- background subtraction (OpenCV's MOG2) is enough to find "the thing
that's different from the empty room" frame by frame, and its centroid is a
usable proxy for the person's position.

This is a first pass to check feasibility, not a finished pipeline: no lens
undistortion, no pixel->real-world calibration, no multi-person handling.

Run:
    .venv/bin/python scripts/extract_trajectory_from_video.py <path/to/video.mp4>
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

VIDEO_PATH = Path(sys.argv[1])
OUT_DIR = Path(__file__).parent.parent / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO_PATH))
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"{VIDEO_PATH.name}: {n_frames} frames at {fps:.2f} fps")

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=True)

positions = []  # (frame_index, t_seconds, x_px, y_px, blob_area) or NaNs if nothing detected
frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    fg_mask = bg_subtractor.apply(frame)
    fg_mask = (fg_mask == 255).astype(np.uint8) * 255  # drop MOG2's "shadow" pixels (value 127)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t = frame_idx / fps

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > 800:  # ignore tiny noise blobs
            M = cv2.moments(largest)
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            positions.append((frame_idx, t, cx, cy, area))
        else:
            positions.append((frame_idx, t, np.nan, np.nan, area))
    else:
        positions.append((frame_idx, t, np.nan, np.nan, 0))

    frame_idx += 1

cap.release()

positions = np.array(positions)
frame_i, t, x, y, area = positions.T
detected_frac = np.mean(~np.isnan(x))
print(f"person detected in {detected_frac:.1%} of frames")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(x, y, "-", alpha=0.4, linewidth=1)
axes[0].scatter(x, y, c=t, cmap="viridis", s=4)
axes[0].invert_yaxis()  # image y-axis grows downward
axes[0].set_xlabel("x (pixels)")
axes[0].set_ylabel("y (pixels)")
axes[0].set_title("pixel-space path (color = time)")

axes[1].plot(t, x, label="x")
axes[1].plot(t, y, label="y")
axes[1].set_xlabel("time (s)")
axes[1].set_ylabel("position (pixels)")
axes[1].set_title("position vs time")
axes[1].legend()

axes[2].plot(t, area)
axes[2].set_xlabel("time (s)")
axes[2].set_ylabel("detected blob area (pixels^2)")
axes[2].set_title("detection confidence over time\n(0 = nobody detected)")

fig.suptitle(f"Background-subtraction trajectory prototype -- {VIDEO_PATH.name}")
fig.tight_layout()
out_path = OUT_DIR / "07_video_trajectory_prototype.png"
fig.savefig(out_path, dpi=130)
print(f"wrote {out_path}")
