"""Visual ground-truth verification: overlays BOTH the old (naive
~isnan) and new (classify_presence_from_gaps) presence labels on the
source video, side by side, so a human can directly watch the exact
frames where they disagree and judge which one is actually correct --
the most direct verification possible, more convincing than any
statistical argument about gap position/length.

Run:
    .venv/bin/python scripts/render_ground_truth_check.py --trial NLoS_EN-W_capture2
Output: results/videos/ground_truth_check_{trial}.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_csi_tracking.data.presence_position_dataset import resample_to_n
from snn_csi_tracking.data.raw_capture_loader import parse_csi_file
from snn_csi_tracking.data.trajectory_extraction import classify_presence_from_gaps
from snn_csi_tracking.training.train_presence_position_conv import RAW_ROOT, TRAJECTORY_CACHE_DIR

OUT_DIR = REPO_ROOT / "results" / "videos"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", default="NLoS_EN-W_capture2")
    args = parser.parse_args()

    condition, activity, capture_name = args.trial.split("_", 2)
    csi_path = RAW_ROOT / f"{condition}_{activity}" / f"{capture_name}.mat"
    segment_idx = capture_name.replace("capture", "")
    video_path = RAW_ROOT / f"{condition}_{activity}" / "videos" / f"segment{segment_idx}.mp4"
    if not video_path.exists():
        raise SystemExit(f"No video at {video_path}")

    print(f"Loading {csi_path}...")
    csi, _timestamps = parse_csi_file(csi_path)
    n_frames = csi.shape[2]

    traj = np.load(TRAJECTORY_CACHE_DIR / f"{args.trial}.npy")
    positions = resample_to_n(traj, n_frames)
    old_present = ~np.isnan(positions[:, 0])
    new_present = classify_presence_from_gaps(positions)
    disagree = old_present != new_present
    print(f"old present rate: {old_present.mean():.3f}   new present rate: {new_present.mean():.3f}   "
          f"frames where they disagree: {disagree.sum()} ({disagree.mean()*100:.1f}%)")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"ground_truth_check_{args.trial}.mp4"
    raw_path = Path(tempfile.mktemp(suffix=".mp4"))
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    csi_frame_per_video_frame = np.linspace(0, n_frames - 1, n_video_frames).astype(int)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f = csi_frame_per_video_frame[frame_idx]
        old_p, new_p = bool(old_present[f]), bool(new_present[f])
        is_disagreement = old_p != new_p

        old_color = (0, 0, 255) if old_p else (150, 150, 150)
        new_color = (0, 255, 0) if new_p else (150, 150, 150)
        old_label = "OLD present" if old_p else "OLD absent"
        new_label = "NEW present" if new_p else "NEW absent"

        cv2.putText(frame, old_label, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, old_color, 2)
        cv2.putText(frame, new_label, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, new_color, 2)
        if is_disagreement:
            cv2.putText(frame, "*** LABELS DISAGREE HERE ***", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 255), 2)
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 255), 6)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-i", str(raw_path), "-c:v", "libx264", "-profile:v", "main",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
        check=True, capture_output=True,
    )
    raw_path.unlink()
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
