"""Quick spy-cam overlay using the simple 2-feature (amp_dev + motion)
logistic-regression presence classifier, properly held out (trained on the
same condition's OTHER activities, excluding whichever trial is being
rendered) -- not the deep conv+SNN model, just a fast visual sanity check
of what that classifier actually does on a genuinely unseen activity.

No position output (this classifier doesn't predict one) -- draws a
generic centered marker when presence is detected, not a tracked blob.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from sklearn.linear_model import LogisticRegression

from snn_csi_tracking.data.presence_position_dataset import compute_features, load_or_build_perframe_dataset
from snn_csi_tracking.data.raw_capture_loader import load_empty_room_baseline, parse_csi_file
from snn_csi_tracking.inference import debounce_presence
from snn_csi_tracking.training.train_presence_position_conv import (
    CACHE_DIR,
    DEPTH_CACHE_DIR,
    RATE_MS,
    RAW_ROOT,
    STRIDE,
    T_WIN,
    TRAJECTORY_CACHE_DIR,
    USE_EMPTY_BASELINE,
    USE_MOTION_MAGNITUDE,
    USE_PHASE,
    session_key,
)

OUT_DIR = Path(__file__).parent.parent / "results" / "videos"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", default="NLoS_EN-W_capture1")
    parser.add_argument("--min-run", type=int, default=6)
    args = parser.parse_args()

    condition, activity, capture_name = args.trial.split("_", 2)
    held_out_session = f"{condition}_{activity}"

    print("Building/loading full dataset for classifier fitting...")
    X, pos, present, groups, activity_codes = load_or_build_perframe_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, DEPTH_CACHE_DIR, CACHE_DIR,
        rate_ms=RATE_MS, t_win=T_WIN, stride=STRIDE, use_3d=False, use_phase=USE_PHASE,
        use_empty_baseline=USE_EMPTY_BASELINE, denoise="wavelet", use_motion_magnitude=USE_MOTION_MAGNITUDE,
    )
    X = np.nan_to_num(X, nan=0.0)
    amp_dev = np.abs(X[:, 0:3, :, :]).mean(axis=(1, 2))
    motion = X[:, 7, :, :].mean(axis=1)
    feat = np.stack([amp_dev, motion], axis=-1)

    sessions = np.array([session_key(g) for g in groups])
    conditions = np.array([g.split("_")[0] for g in groups])
    train_mask = (conditions == condition) & (sessions != held_out_session)
    Xf, yf = feat.reshape(-1, 2), present.reshape(-1)
    train_idx = np.repeat(train_mask, T_WIN)
    print(f"Training on {condition} sessions excluding {held_out_session} "
          f"({train_mask.sum()} windows, {len(set(sessions[train_mask]))} sessions)...")
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xf[train_idx], yf[train_idx])

    csi_path = RAW_ROOT / f"{condition}_{activity}" / f"{capture_name}.mat"
    segment_idx = capture_name.replace("capture", "")
    video_path = RAW_ROOT / f"{condition}_{activity}" / "videos" / f"segment{segment_idx}.mp4"
    if not video_path.exists():
        raise SystemExit(f"No video at {video_path}")

    print(f"Loading {csi_path}...")
    csi, _timestamps = parse_csi_file(csi_path)
    n_frames = csi.shape[2]
    baseline = load_empty_room_baseline(RAW_ROOT, condition, denoise="wavelet")
    amp_z = compute_features(csi, use_phase=USE_PHASE, baseline=baseline, denoise="wavelet", use_motion_magnitude=USE_MOTION_MAGNITUDE)
    amp_dev_t = np.abs(amp_z[0:3]).mean(axis=(0, 1))
    motion_t = amp_z[7].mean(axis=0)
    feat_t = np.stack([amp_dev_t, motion_t], axis=-1)
    probs = clf.predict_proba(feat_t)[:, 1]
    presence_binary = debounce_presence(probs, threshold=0.5, min_run=args.min_run)
    print(f"fraction detected present: {presence_binary.mean():.2f} (held-out, never trained on this session)")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"spy_overlay_simple_{args.trial}.mp4"
    raw_path = Path(tempfile.mktemp(suffix=".mp4"))
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    csi_frame_per_video_frame = np.linspace(0, n_frames - 1, n_video_frames).astype(int)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f = csi_frame_per_video_frame[frame_idx]
        is_present = bool(presence_binary[f])
        if is_present:
            overlay = frame.copy()
            cv2.circle(overlay, (w // 2, h // 2), 60, (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        label = f"presence: {probs[f]:.2f}" + ("  [DETECTED]" if is_present else "  [empty]")
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
