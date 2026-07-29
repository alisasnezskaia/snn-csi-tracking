"""Ground-truth vs predicted trajectory for one trial, each plotted as a
connected line through its points in chronological order. 3D (X, Y, Z) in
meters (camera frame -- see camera_geometry.deproject_pixel_to_camera_frame)
for per-frame 3D checkpoints, 2D (x, y) normalized image-plane otherwise.

"Ground truth" (X, Y, Z) is reconstructed the same way the 3D model's
training labels were built: resample_depth (metric depth, meters) then
deproject_pixel_to_camera_frame -- not an independently measured value, but
the same real-world deprojection used at training time, not a relative/
arbitrary-unit depth (see depth_extraction.py's docstring).

Overlapping window predictions are stitched into one continuous per-frame
stream by averaging, same idea as the notebook's spy-cam overlay.

Run:
    .venv/bin/python scripts/visualize_trajectory_prediction.py
    .venv/bin/python scripts/visualize_trajectory_prediction.py --trial NLoS_S_capture2 --kind windowed
    .venv/bin/python scripts/visualize_trajectory_prediction.py --kind perframe3d
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from snn_csi_tracking.data.camera_geometry import deproject_pixel_to_camera_frame
from snn_csi_tracking.data.depth_extraction import depth_cache_path, get_video_dimensions
from snn_csi_tracking.data.presence_position_dataset import compute_features, resample_depth, resample_to_n
from snn_csi_tracking.data.raw_capture_loader import discover_captures, parse_csi_file
from snn_csi_tracking.inference import DEFAULT_CHECKPOINTS, load_model, predict_dense

REPO_ROOT = Path(__file__).parent.parent
RAW_ROOT = REPO_ROOT / "data" / "raw_captures"
TRAJECTORY_CACHE_DIR = REPO_ROOT / "data" / "processed" / "presence_position_trajectories"
DEPTH_CACHE_DIR = REPO_ROOT / "data" / "processed" / "presence_position_depth"
OUT_PATH = REPO_ROOT / "results" / "figures" / "14_trajectory_prediction.png"

RATE_MS = 50
PRESENCE_THRESHOLD = 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial", default="PLoS_L_capture5")
    parser.add_argument("--kind", choices=["windowed", "perframe2d", "perframe3d"], default="perframe2d")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINTS[args.kind]

    condition, activity, capture_name = args.trial.split("_", 2)
    csi_path = RAW_ROOT / f"{condition}_{activity}" / f"{capture_name}.mat"
    traj_path = TRAJECTORY_CACHE_DIR / f"{args.trial}.npy"

    print(f"Loading {csi_path}...")
    csi, _timestamps = parse_csi_file(csi_path)
    n_frames = csi.shape[2]

    trajectory = np.load(traj_path)
    gt_positions = resample_to_n(trajectory, n_frames)
    gt_present = ~np.isnan(gt_positions[:, 0])

    print(f"Loading {checkpoint_path} ({args.kind})...")
    model, use_phase = load_model(args.kind, checkpoint_path)
    amp_z = compute_features(csi, use_phase=use_phase)
    pred_presence, pred_position, pred_valid = predict_dense(model, args.kind, amp_z)
    pred_present = pred_valid & (pred_presence >= PRESENCE_THRESHOLD)
    is_3d = pred_position.shape[-1] == 3

    if is_3d:
        depth_path = depth_cache_path(DEPTH_CACHE_DIR, args.trial)
        if not depth_path.exists():
            raise SystemExit(f"No cached depth for {args.trial} at {depth_path} -- can't build the (X,Y,Z) ground truth.")
        matching_captures = [c for c in discover_captures(RAW_ROOT) if c.trial_key == args.trial]
        if not matching_captures:
            raise SystemExit(f"No capture found for trial {args.trial} under {RAW_ROOT}")
        width, height = get_video_dimensions(matching_captures[0].video_path)
        depth_raw = np.load(depth_path)
        gt_z = resample_depth(depth_raw, n_frames)
        gt_positions = deproject_pixel_to_camera_frame(gt_positions, gt_z, width, height)
        gt_present = gt_present & ~np.isnan(gt_z)

    print(f"ground truth: {gt_present.sum()}/{n_frames} frames present")
    print(f"predicted:    {pred_present.sum()}/{n_frames} frames present")

    time_s = np.arange(n_frames) * RATE_MS / 1000

    if is_3d:
        fig = plt.figure(figsize=(20, 6))
        ax3d = fig.add_subplot(1, 4, 1, projection="3d")
        ax3d.plot(gt_positions[gt_present, 0], gt_positions[gt_present, 1], gt_positions[gt_present, 2],
                  "-o", color="tab:blue", markersize=2, linewidth=1, alpha=0.8, label="ground truth")
        ax3d.plot(pred_position[pred_present, 0], pred_position[pred_present, 1], pred_position[pred_present, 2],
                  "-o", color="tab:orange", markersize=2, linewidth=1, alpha=0.8, label="predicted")
        ax3d.set_xlabel("X (m, camera frame)")
        ax3d.set_ylabel("Y (m, camera frame)")
        ax3d.set_zlabel("Z (m, camera frame)")
        ax3d.set_title(f"{args.trial}: 3D trajectory ({args.kind})")
        ax3d.invert_yaxis()
        ax3d.legend()
        time_axes = [fig.add_subplot(1, 4, i + 2) for i in range(3)]
        dims = [(0, "x"), (1, "y"), (2, "z")]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        ax = axes[0]
        ax.plot(gt_positions[gt_present, 0], gt_positions[gt_present, 1], "-o", color="tab:blue",
                markersize=2, linewidth=1, alpha=0.8, label="ground truth")
        ax.plot(pred_position[pred_present, 0], pred_position[pred_present, 1], "-o", color="tab:orange",
                markersize=2, linewidth=1, alpha=0.8, label="predicted")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{args.trial}: trajectory ({args.kind})")
        ax.invert_yaxis()  # image coords: y grows downward
        ax.legend()
        time_axes = axes[1:]
        dims = [(0, "x"), (1, "y")]

    for ax, (dim, label) in zip(time_axes, dims):
        ax.plot(time_s[gt_present], gt_positions[gt_present, dim], "-o", color="tab:blue", markersize=2, label="ground truth")
        ax.plot(time_s[pred_present], pred_position[pred_present, dim], "-o", color="tab:orange", markersize=2, label="predicted")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(label)
        ax.set_title(f"{label}(t)")
        ax.legend()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
