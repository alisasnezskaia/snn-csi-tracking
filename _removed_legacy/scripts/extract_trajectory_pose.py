"""Extract a trajectory from a static-camera video using pose detection instead
of background subtraction, tracking the hip midpoint (stable across standing/
sitting/bending, unlike a raw silhouette centroid -- see conversation notes).

Same output format as extract_trajectory_from_video.py for direct comparison.

Run:
    .venv/bin/python scripts/extract_trajectory_pose.py <path/to/video.mp4>
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

VIDEO_PATH = Path(sys.argv[1])
OUT_DIR = Path(__file__).parent.parent / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mp_pose = mp.solutions.pose
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP
RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP

cap = cv2.VideoCapture(str(VIDEO_PATH))
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"{VIDEO_PATH.name}: {n_frames} frames at {fps:.2f} fps, {w}x{h}")

positions = []  # (frame_idx, t, x_px, y_px, confidence)
frame_idx = 0
with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            left, right = lm[LEFT_HIP], lm[RIGHT_HIP]
            conf = (left.visibility + right.visibility) / 2
            if conf > 0.5:
                x = (left.x + right.x) / 2 * w
                y = (left.y + right.y) / 2 * h
                positions.append((frame_idx, t, x, y, conf))
            else:
                positions.append((frame_idx, t, np.nan, np.nan, conf))
        else:
            positions.append((frame_idx, t, np.nan, np.nan, 0.0))

        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  ...{frame_idx}/{n_frames} frames")

cap.release()

positions = np.array(positions)
frame_i, t, x, y, conf = positions.T
detected_frac = np.mean(~np.isnan(x))
print(f"hip detected (confidence>0.5) in {detected_frac:.1%} of frames")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(x, y, "-", alpha=0.4, linewidth=1)
axes[0].scatter(x, y, c=t, cmap="viridis", s=4)
axes[0].invert_yaxis()
axes[0].set_xlabel("x (pixels)")
axes[0].set_ylabel("y (pixels)")
axes[0].set_title("hip-midpoint path (color = time)")

axes[1].plot(t, x, label="x")
axes[1].plot(t, y, label="y")
axes[1].set_xlabel("time (s)")
axes[1].set_ylabel("position (pixels)")
axes[1].set_title("position vs time")
axes[1].legend()

axes[2].plot(t, conf)
axes[2].axhline(0.5, color="red", linestyle="--", linewidth=1, label="confidence threshold")
axes[2].set_xlabel("time (s)")
axes[2].set_ylabel("hip landmark confidence")
axes[2].set_title("detection confidence over time")
axes[2].legend()

fig.suptitle(f"Pose-based hip-midpoint trajectory -- {VIDEO_PATH.name}")
fig.tight_layout()
out_path = OUT_DIR / "08_video_trajectory_pose.png"
fig.savefig(out_path, dpi=130)
print(f"wrote {out_path}")
