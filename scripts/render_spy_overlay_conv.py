"""Same "spy-cam" overlay video as render_spy_overlay.py, but for the new
conv+SNN continuous position model (SNNPresencePositionConvPerFrame,
train_presence_position_conv.py) instead of the older flat-MLP checkpoints
render_spy_overlay.py targets -- that model takes raw per-frame features
directly (does its own conv + delta/spike encoding internally), so it needs
its own dense-prediction stitching (inference.predict_dense_conv) rather
than render_spy_overlay.py's to_spikes-then-model call.

Reuses train_presence_position_conv.py's own architecture/feature constants
(CONV_CHANNELS, KERNEL_SIZE, USE_EMPTY_BASELINE, etc.) so this always
matches whatever that script actually trained, instead of duplicating and
risking drift.

Run:
    .venv/bin/python scripts/render_spy_overlay_conv.py
    .venv/bin/python scripts/render_spy_overlay_conv.py --trial NLoS_S_capture2 --out-dim 3 \
        --checkpoint results/models/presence_position_conv_3d_wavelet_50ms.pt
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch

from snn_csi_tracking.data.presence_position_dataset import compute_features
from snn_csi_tracking.data.raw_capture_loader import load_empty_room_baseline, parse_csi_file
from snn_csi_tracking.inference import DEVICE, debounce_presence, predict_dense_conv, smooth_position
from snn_csi_tracking.models.presence_position_snn import SNNPresencePositionConvPerFrame
from snn_csi_tracking.training.train_presence_position_conv import (
    AMPLITUDE_NORM,
    CONV_CHANNELS,
    DELTA_THRESHOLD,
    HIDDEN_1,
    HIDDEN_2,
    KERNEL_SIZE,
    RAW_ROOT,
    USE_EMPTY_BASELINE,
    USE_MOTION_MAGNITUDE,
    USE_PHASE,
)

OUT_DIR = Path(__file__).parent.parent / "results" / "videos"
DEFAULT_CHECKPOINT = Path(__file__).parent.parent / "results" / "models" / "presence_position_conv_2d_wavelet_50ms.pt"

PRESENCE_THRESHOLD = 0.5
DEBOUNCE_MIN_RUN = 6
SMOOTH_WINDOW = 5
BLOB_RADIUS = 45


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial", default="PLoS_L_capture5")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--denoise", choices=["none", "wavelet", "pca"], default="wavelet")
    parser.add_argument("--out-dim", type=int, default=2, choices=[2, 3])
    parser.add_argument("--use-cross-coherence", action="store_true",
                         help="build the cross-antenna-coherence channel too -- required to match a "
                              "checkpoint trained with feature=cross_coherence (current best full-pipeline "
                              "config, AUROC=0.608+/-0.028); channel count must match training exactly")
    parser.add_argument("--presence-threshold", type=float, default=PRESENCE_THRESHOLD)
    parser.add_argument("--min-run", type=int, default=DEBOUNCE_MIN_RUN,
                         help="consecutive frames required before presence flips state (debounce strength)")
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW,
                         help="centered rolling-mean window (frames) applied to position within each presence stretch")
    args = parser.parse_args()
    denoise = None if args.denoise == "none" else args.denoise

    condition, activity, capture_name = args.trial.split("_", 2)
    csi_path = RAW_ROOT / f"{condition}_{activity}" / f"{capture_name}.mat"
    segment_idx = capture_name.replace("capture", "")
    video_path = RAW_ROOT / f"{condition}_{activity}" / "videos" / f"segment{segment_idx}.mp4"

    if not video_path.exists():
        raise SystemExit(f"No video for {args.trial} at {video_path} -- can't render an overlay without it.")

    print(f"Loading {csi_path}...")
    csi, _timestamps = parse_csi_file(csi_path)
    n_frames = csi.shape[2]

    baseline = load_empty_room_baseline(RAW_ROOT, condition, denoise=denoise) if USE_EMPTY_BASELINE else None
    amp_z = compute_features(
        csi, use_phase=USE_PHASE, baseline=baseline, denoise=denoise, use_motion_magnitude=USE_MOTION_MAGNITUDE,
        amplitude_norm=AMPLITUDE_NORM, use_cross_coherence=args.use_cross_coherence,
    )

    print(f"Loading {args.checkpoint}...")
    model = SNNPresencePositionConvPerFrame(
        num_channels=amp_z.shape[0], num_subcarriers=amp_z.shape[1], conv_channels=CONV_CHANNELS,
        h1=HIDDEN_1, h2=HIDDEN_2, out_dim=args.out_dim, kernel_size=KERNEL_SIZE, delta_threshold=DELTA_THRESHOLD,
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    presence_prob, position, valid = predict_dense_conv(model, amp_z)
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
    out_path = OUT_DIR / f"spy_overlay_conv_{args.trial}_{args.denoise}.mp4"

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
