"""Renders a "spy-cam" overlay video: model-predicted presence + position,
drawn as a red blob on top of the actual trial video.

Ported from the notebook's build_spy_overlay / build_spy_overlay_perframe,
with deliberate changes: presence is always debounced (hysteresis over
`--min-run` consecutive frames) before being drawn, never thresholded
frame-by-frame -- the raw per-frame signal flickers visibly (see
conversation), so an un-debounced version isn't offered here. Position is
likewise smoothed (rolling mean over `--smooth-window` frames, within each
debounced presence stretch) before being drawn, since the raw per-window
position prediction jumps frame to frame.

Frames are drawn via cv2.VideoWriter (needs per-frame compositing), then
re-encoded to H.264 with ffmpeg -- cv2's mp4v output isn't decodable by
browsers or most players.

Run:
    .venv/bin/python scripts/render_spy_overlay.py
    .venv/bin/python scripts/render_spy_overlay.py --trial NLoS_S_capture2 --kind windowed
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from snn_csi_tracking.data.presence_position_dataset import compute_features
from snn_csi_tracking.data.raw_capture_loader import parse_csi_file
from snn_csi_tracking.inference import DEFAULT_CHECKPOINTS, debounce_presence, load_model, predict_dense, smooth_position

REPO_ROOT = Path(__file__).parent.parent
RAW_ROOT = REPO_ROOT / "data" / "raw_captures"
OUT_DIR = REPO_ROOT / "results" / "videos"

PRESENCE_THRESHOLD = 0.5
DEBOUNCE_MIN_RUN = 6
SMOOTH_WINDOW = 5
BLOB_RADIUS = 45


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial", default="PLoS_L_capture5")
    parser.add_argument("--kind", choices=["windowed", "perframe2d", "perframe3d"], default="perframe3d")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--presence-threshold", type=float, default=PRESENCE_THRESHOLD)
    parser.add_argument("--min-run", type=int, default=DEBOUNCE_MIN_RUN,
                         help="consecutive frames required before presence flips state (debounce strength)")
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW,
                         help="centered rolling-mean window (frames) applied to position within each presence stretch")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINTS[args.kind]
    condition, activity, capture_name = args.trial.split("_", 2)
    csi_path = RAW_ROOT / f"{condition}_{activity}" / f"{capture_name}.mat"
    segment_idx = capture_name.replace("capture", "")
    video_path = RAW_ROOT / f"{condition}_{activity}" / "videos" / f"segment{segment_idx}.mp4"

    if not video_path.exists():
        raise SystemExit(f"No video for {args.trial} at {video_path} -- can't render an overlay without it.")

    print(f"Loading {csi_path}...")
    csi, _timestamps = parse_csi_file(csi_path)
    n_frames = csi.shape[2]

    print(f"Loading {checkpoint_path} ({args.kind})...")
    model, use_phase = load_model(args.kind, checkpoint_path)
    amp_z = compute_features(csi, use_phase=use_phase)
    presence_prob, position, valid = predict_dense(model, args.kind, amp_z)
    presence_prob_filled = np.nan_to_num(presence_prob, nan=0.0)
    presence_binary = debounce_presence(presence_prob_filled, threshold=args.presence_threshold, min_run=args.min_run)
    present = presence_binary & valid
    position = smooth_position(position, present, window=args.smooth_window)
    print(f"debounced presence: {presence_binary.mean():.2f} fraction of frames DETECTED "
          f"(min_run={args.min_run}, threshold={args.presence_threshold})")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"spy_overlay_{args.trial}_{args.kind}.mp4"

    # cv2.VideoWriter needs a codec its OpenCV build actually ships an encoder
    # for -- opencv-python-headless doesn't bundle libx264, so writing "avc1"
    # directly here fails silently. mp4v (MPEG-4 Part 2) always works but
    # isn't decodable by browsers/most players, so we draw frames into a
    # throwaway mp4v file, then have ffmpeg re-encode that to real H.264 for
    # the file we actually keep.
    raw_path = Path(tempfile.mktemp(suffix=".mp4"))
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    csi_frame_per_video_frame = np.linspace(0, n_frames - 1, n_video_frames).astype(int)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f = csi_frame_per_video_frame[frame_idx]
        is_present = bool(present[f])
        x, y = position[f, 0], position[f, 1]

        if is_present and not np.isnan(x):
            overlay = frame.copy()
            px, py = int(np.clip(x, 0, 1) * w), int(np.clip(y, 0, 1) * h)
            cv2.circle(overlay, (px, py), BLOB_RADIUS, (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        label = f"presence: {presence_prob_filled[f]:.2f}" + ("  [DETECTED]" if is_present else "  [empty]")
        cv2.putText(frame, label, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if is_present else (150, 150, 150), 2)
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
