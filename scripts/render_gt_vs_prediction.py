"""Ground truth vs model prediction, presence only -- overlays BOTH on the
source video side by side (same "two labels + disagreement highlight"
convention as render_ground_truth_check.py, but comparing the trained
model's presence output against ground truth instead of two labeling
schemes against each other).

Built to directly verify the presence-flicker-during-sitting failure mode:
the L activity's ground truth says "present" through the whole sitting
stretch, but every motion-based feature this pipeline uses goes quiet once
the person stops moving, so the model's own prediction should visibly
disagree with ground truth exactly there -- this makes that failure (and
any fix's effect on it) directly watchable rather than just a number in a
CV table.

Run:
    .venv/bin/python scripts/render_gt_vs_prediction.py --trial PLoS_L_capture1 \\
        --checkpoint results/models/presence_position_snn_cross_coherence_demo.pt --use-cross-coherence
Output: results/videos/gt_vs_pred_{trial}.mp4
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
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_csi_tracking.data.presence_position_dataset import compute_features, resample_to_n
from snn_csi_tracking.data.raw_capture_loader import load_empty_room_baseline, parse_csi_file
from snn_csi_tracking.data.trajectory_extraction import classify_presence_from_gaps
from snn_csi_tracking.inference import DEVICE, debounce_presence, predict_dense_conv
from snn_csi_tracking.models.presence_position_snn import SNNPresencePositionConvPerFrame
from snn_csi_tracking.training.train_presence_position_conv import (
    AMPLITUDE_NORM, CONV_CHANNELS, DELTA_THRESHOLD, HIDDEN_1, HIDDEN_2, KERNEL_SIZE,
    RAW_ROOT, TRAJECTORY_CACHE_DIR, USE_MOTION_MAGNITUDE, USE_PHASE,
)

OUT_DIR = REPO_ROOT / "results" / "videos"
PRESENCE_THRESHOLD = 0.5
DEBOUNCE_MIN_RUN = 6


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial", default="PLoS_L_capture1")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dim", type=int, default=2, choices=[2, 3])
    parser.add_argument("--use-cross-coherence", action="store_true",
                         help="must match whatever the checkpoint was trained with -- channel count has to line up")
    parser.add_argument("--use-baseline-deviation", action="store_true",
                         help="must match whatever the checkpoint was trained with -- channel count has to line up")
    parser.add_argument("--presence-threshold", type=float, default=PRESENCE_THRESHOLD)
    parser.add_argument("--min-run", type=int, default=DEBOUNCE_MIN_RUN,
                         help="consecutive frames required before presence flips state (debounce strength)")
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
    gt_present = classify_presence_from_gaps(positions)
    print(f"ground truth present rate: {gt_present.mean():.3f}")

    baseline_deviation = None
    if args.use_baseline_deviation:
        baseline_deviation = load_empty_room_baseline(RAW_ROOT, condition)
        if baseline_deviation is None:
            raise SystemExit(f"--use-baseline-deviation but no empty-room capture found for condition {condition!r}")

    amp_z = compute_features(
        csi, use_phase=USE_PHASE, baseline=None, use_motion_magnitude=USE_MOTION_MAGNITUDE,
        amplitude_norm=AMPLITUDE_NORM, use_cross_coherence=args.use_cross_coherence,
        baseline_deviation=baseline_deviation,
    )

    print(f"Loading {args.checkpoint}...")
    model = SNNPresencePositionConvPerFrame(
        num_channels=amp_z.shape[0], num_subcarriers=amp_z.shape[1], conv_channels=CONV_CHANNELS,
        h1=HIDDEN_1, h2=HIDDEN_2, out_dim=args.out_dim, kernel_size=KERNEL_SIZE, delta_threshold=DELTA_THRESHOLD,
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    presence_prob, _position, valid = predict_dense_conv(model, amp_z)
    presence_prob_filled = np.nan_to_num(presence_prob, nan=0.0)
    pred_present = debounce_presence(presence_prob_filled, threshold=args.presence_threshold, min_run=args.min_run) & valid

    disagree = gt_present != pred_present
    print(f"predicted present rate: {pred_present.mean():.3f}   "
          f"frames where they disagree: {disagree.sum()} ({disagree.mean()*100:.1f}%)")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"gt_vs_pred_{args.trial}.mp4"
    raw_path = Path(tempfile.mktemp(suffix=".mp4"))
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    csi_frame_per_video_frame = np.linspace(0, n_frames - 1, n_video_frames).astype(int)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f = csi_frame_per_video_frame[frame_idx]
        gt_p, pred_p = bool(gt_present[f]), bool(pred_present[f])
        is_disagreement = gt_p != pred_p

        gt_color = (0, 255, 0) if gt_p else (150, 150, 150)
        pred_color = (0, 0, 255) if pred_p else (150, 150, 150)
        gt_label = "GROUND TRUTH: present" if gt_p else "GROUND TRUTH: absent"
        pred_label = f"MODEL: present ({presence_prob_filled[f]:.2f})" if pred_p else f"MODEL: absent ({presence_prob_filled[f]:.2f})"

        cv2.putText(frame, gt_label, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gt_color, 2)
        cv2.putText(frame, pred_label, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, pred_color, 2)
        if is_disagreement:
            cv2.putText(frame, "*** MODEL DISAGREES WITH GROUND TRUTH ***", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
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
