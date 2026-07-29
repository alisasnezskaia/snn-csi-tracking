"""Visualizes the actual model input features over time, one row per
SESSION (condition x activity -- 10 total: both NLoS and PLoS, all 5
activities, paired adjacently for direct comparison) -- answers "what are
we actually feeding the model, and how do sessions differ from each other"
directly, rather than by inference from aggregate accuracy numbers. This is
the same-activity-different-condition comparison directly relevant to the
diagnosed cross-session generalization failure (AUROC~0.508 across every
leave-activity-out fold, see conversation) -- if two sessions of the SAME
activity look meaningfully different in absolute scale, that's the
session-fingerprint-overfitting mechanism made visible.

Per session (one representative capture, capture1), plots (all from the
SAME feature pipeline the models actually train on --
presence_position_dataset.compute_features / motion_magnitude_feature, not
a separate ad-hoc computation):
  1. Ground-truth presence (shaded region) against the resampled trajectory
  2. Per-antenna amplitude, subcarrier-averaged (3 lines) -- current
     AMPLITUDE_NORM (energy or zscore, see train_presence_position_conv.py)
  3. One representative phase-difference channel (sin, antenna 1 vs ref),
     subcarrier-averaged
  4. Motion-magnitude feature (already a scalar-per-frame quantity)

Run:
    .venv/bin/python scripts/plot_features_per_activity.py
Output: results/figures/features_per_activity.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_csi_tracking.data.presence_position_dataset import compute_features, motion_magnitude_feature, resample_to_n
from snn_csi_tracking.data.raw_capture_loader import parse_csi_file
from snn_csi_tracking.data.trajectory_extraction import classify_presence_from_gaps, load_or_extract_trajectory
from snn_csi_tracking.training.train_presence_position_conv import (
    AMPLITUDE_NORM, RAW_ROOT, TRAJECTORY_CACHE_DIR, USE_MOTION_MAGNITUDE, USE_PHASE,
)

ACTIVITIES = ["E", "EN-S", "EN-W", "L", "S"]
CONDITIONS = ["NLoS", "PLoS"]
CAPTURE = "capture1"
SESSIONS = [(activity, condition) for activity in ACTIVITIES for condition in CONDITIONS]  # paired rows


def load_one(condition: str, activity: str):
    csi_path = RAW_ROOT / f"{condition}_{activity}" / f"{CAPTURE}.mat"
    csi, _timestamps = parse_csi_file(csi_path)  # (Ant, Sub, T)
    trial_key = f"{condition}_{activity}_{CAPTURE}"
    cache_path = TRAJECTORY_CACHE_DIR / f"{trial_key}.npy"
    trajectory = np.load(cache_path) if cache_path.exists() else None
    if trajectory is None:
        raise FileNotFoundError(f"no cached trajectory for {trial_key} -- run the dataset builder first")
    positions = resample_to_n(trajectory, csi.shape[2])
    present = classify_presence_from_gaps(positions).astype(np.float32)

    feat = compute_features(csi, use_phase=USE_PHASE, use_motion_magnitude=USE_MOTION_MAGNITUDE,
                             amplitude_norm=AMPLITUDE_NORM)  # (Chan, Sub, T)
    motion = motion_magnitude_feature(csi)[0].mean(axis=0)  # (T,) -- already broadcast across subcarriers
    raw_amp_mean = np.abs(csi).mean()  # un-normalized, for cross-session absolute-scale comparison
    return feat, present, motion, raw_amp_mean


def main():
    n = len(SESSIONS)
    fig, axes = plt.subplots(n, 3, figsize=(15, 2.4 * n), sharex=False)
    fig.suptitle(f"Per-session feature comparison (capture1, amplitude_norm={AMPLITUDE_NORM}) "
                 f"-- rows paired NLoS/PLoS per activity", fontsize=13)

    for row, (activity, condition) in enumerate(SESSIONS):
        feat, present, motion, raw_amp_mean = load_one(condition, activity)
        t = np.arange(feat.shape[-1])
        num_antennas = 3  # see preprocessing.num_feature_channels: 3 amp + 4 phase-diff (sin,cos)x2
        label = f"{condition}_{activity}"

        ax_amp, ax_phase, ax_motion = axes[row]

        for a in range(num_antennas):
            amp_trace = feat[a].mean(axis=0)  # subcarrier-averaged amplitude, antenna a
            ax_amp.plot(t, amp_trace, label=f"ant{a}", linewidth=0.8)
        ax_amp.fill_between(t, ax_amp.get_ylim()[0], ax_amp.get_ylim()[1], where=present > 0,
                             color="green", alpha=0.08, step="mid")
        ax_amp.set_title(f"{label}: amplitude (raw mean={raw_amp_mean:.4f})")
        if row == 0:
            ax_amp.legend(fontsize=7, loc="upper right")

        # phase-diff channel index num_antennas (first sin channel, antenna 1 vs ref)
        phase_trace = feat[num_antennas].mean(axis=0)
        ax_phase.plot(t, phase_trace, color="purple", linewidth=0.8)
        ax_phase.fill_between(t, -1, 1, where=present > 0, color="green", alpha=0.08, step="mid")
        ax_phase.set_ylim(-1, 1)
        ax_phase.set_title(f"{label}: phase-diff sin(ant1-ant0)")

        ax_motion.plot(t, motion, color="darkorange", linewidth=0.8)
        ax_motion.fill_between(t, 0, motion.max() * 1.05 if motion.max() > 0 else 1, where=present > 0,
                                color="green", alpha=0.08, step="mid")
        ax_motion.set_title(f"{label}: motion magnitude")

        for ax in (ax_amp, ax_phase, ax_motion):
            ax.set_xlabel("frame")
        print(f"{label}: presence_rate={present.mean():.3f}, raw_amp_mean={raw_amp_mean:.5f}, "
              f"motion_mean={motion.mean():.4f}, motion_max={motion.max():.4f}")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = REPO_ROOT / "results" / "figures" / "features_per_activity.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
