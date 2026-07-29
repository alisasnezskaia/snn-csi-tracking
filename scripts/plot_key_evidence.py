"""Compact, paper-ready figure isolating the two clearest pieces of
diagnostic evidence from tonight's investigation (see conversation) -- a
curated subset of results/figures/features_per_activity.png (which remains
available in full, 10-session form as supplementary material):

  1. NLoS_E vs PLoS_E: two sessions of the SAME activity (empty room, 100%
     ground-truth absent in both) show a >4x difference in raw motion-
     magnitude noise floor -- direct evidence that no single absolute
     threshold can separate "occupied" from "empty" across sessions.
  2. NLoS_L vs PLoS_L: after the person leaves (right edge of the shaded
     region), the signal does NOT settle back to that session's own
     empty-room reference level -- confirmed quantitatively in
     conversation (motion magnitude settles BELOW the NLoS empty reference
     but ABOVE the PLoS one -- opposite directions, same activity).

Run:
    .venv/bin/python scripts/plot_key_evidence.py
Output: results/figures/key_evidence.png
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
from snn_csi_tracking.training.train_presence_position_conv import (
    AMPLITUDE_NORM, RAW_ROOT, TRAJECTORY_CACHE_DIR, USE_MOTION_MAGNITUDE, USE_PHASE,
)

SESSIONS = [("NLoS", "E"), ("PLoS", "E"), ("NLoS", "L"), ("PLoS", "L")]
CAPTURE = "capture1"


def load_one(condition: str, activity: str):
    csi_path = RAW_ROOT / f"{condition}_{activity}" / f"{CAPTURE}.mat"
    csi, _timestamps = parse_csi_file(csi_path)
    trial_key = f"{condition}_{activity}_{CAPTURE}"
    trajectory = np.load(TRAJECTORY_CACHE_DIR / f"{trial_key}.npy")
    positions = resample_to_n(trajectory, csi.shape[2])
    present = (~np.isnan(positions[:, 0])).astype(np.float32)
    feat = compute_features(csi, use_phase=USE_PHASE, use_motion_magnitude=USE_MOTION_MAGNITUDE,
                             amplitude_norm=AMPLITUDE_NORM)
    motion = motion_magnitude_feature(csi)[0].mean(axis=0)
    return feat, present, motion


def main():
    fig, axes = plt.subplots(len(SESSIONS), 2, figsize=(11, 2.6 * len(SESSIONS)))
    fig.suptitle("Key diagnostic evidence: session-to-session noise-floor drift breaks presence detection\n"
                 "(both plots use the exact model-input feature pipeline, not an ad-hoc computation)",
                 fontsize=12)

    motion_ranges = {}
    for activity in {a for _, a in SESSIONS}:
        motion_ranges[activity] = max(
            load_one(c, activity)[2].max() for c in ("NLoS", "PLoS")
        )

    for row, (condition, activity) in enumerate(SESSIONS):
        feat, present, motion = load_one(condition, activity)
        t = np.arange(feat.shape[-1])
        label = f"{condition}_{activity}"

        ax_amp, ax_motion = axes[row]
        for a in range(3):
            ax_amp.plot(t, feat[a].mean(axis=0), linewidth=0.8, label=f"ant{a}" if row == 0 else None)
        ax_amp.fill_between(t, ax_amp.get_ylim()[0], ax_amp.get_ylim()[1], where=present > 0,
                             color="green", alpha=0.08, step="mid")
        ax_amp.set_title(f"{label}: amplitude")
        if row == 0:
            ax_amp.legend(fontsize=7, loc="upper right")

        ymax = motion_ranges[activity] * 1.05
        ax_motion.plot(t, motion, color="darkorange", linewidth=0.8)
        ax_motion.fill_between(t, 0, ymax, where=present > 0, color="green", alpha=0.08, step="mid")
        ax_motion.set_ylim(0, ymax)  # SAME y-scale within an activity pair -- makes the noise-floor gap visually honest
        ax_motion.set_title(f"{label}: motion magnitude (mean={motion.mean():.1f})")
        ax_amp.set_xlabel("frame")
        ax_motion.set_xlabel("frame")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = REPO_ROOT / "results" / "figures" / "key_evidence.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
