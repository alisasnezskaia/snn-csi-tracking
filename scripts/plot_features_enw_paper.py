"""Paper-ready feature crop, extended to pair two activities: EN-W (person
enters, walks -- continuous obvious motion) and L (person sits, then stands
and leaves -- long stationary stretch with ground-truth presence, then real
motion at the end). One representative condition (PLoS) per activity, so
the figure stays compact (2 rows x 2 columns) while directly motivating two
separate discussion points: (i) EN-W looks visually unambiguous yet is the
hardest fold to generalize to, (ii) L's sitting stretch shows ground truth
"present" during a period where every motion-based feature goes quiet --
the failure mode motivating the delta-encoding architecture discussion.
Real LaTeX text rendering (matplotlib text.usetex) to match the paper's
fonts exactly.

Run:
    .venv/bin/python scripts/plot_features_enw_paper.py
Output: results/figures/features_enw_paper.png (and .pdf)
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
from snn_csi_tracking.data.trajectory_extraction import classify_presence_from_gaps
from snn_csi_tracking.training.train_presence_position_conv import (
    AMPLITUDE_NORM, RAW_ROOT, TRAJECTORY_CACHE_DIR, USE_MOTION_MAGNITUDE, USE_PHASE, RATE_MS,
)

FIGWIDTH_IN = 11.0  # canvas is ~2x wider than the other paper figures (4.5-5.5in) -- font
# sizes below are scaled up proportionally (~2 pt/inch, matching those figures' pt/width
# ratio) so text lands at the same effective size once every figure is scaled to a common
# page width, instead of matching their raw 10pt (which would render smaller here).
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 21,
    "axes.titlesize": 21,
    "axes.labelsize": 21,
    "xtick.labelsize": 21,
    "ytick.labelsize": 19,
})

CONDITION = "PLoS"
CAPTURE = "capture1"
ACTIVITIES = [
    ("EN-W", "enters, walks"),
    ("L", "sits, then stands and leaves"),
]


def load_one(activity: str):
    csi_path = RAW_ROOT / f"{CONDITION}_{activity}" / f"{CAPTURE}.mat"
    csi, _timestamps = parse_csi_file(csi_path)
    trial_key = f"{CONDITION}_{activity}_{CAPTURE}"
    trajectory = np.load(TRAJECTORY_CACHE_DIR / f"{trial_key}.npy")
    positions = resample_to_n(trajectory, csi.shape[2])
    present = classify_presence_from_gaps(positions).astype(np.float32)
    feat = compute_features(csi, use_phase=USE_PHASE, use_motion_magnitude=USE_MOTION_MAGNITUDE,
                             amplitude_norm=AMPLITUDE_NORM)
    motion = motion_magnitude_feature(csi)[0].mean(axis=0)
    return feat, present, motion


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), sharex=False)

    legend_handles, legend_labels = None, None
    for row, (activity, desc) in enumerate(ACTIVITIES):
        feat, present, motion = load_one(activity)
        t_sec = np.arange(feat.shape[-1]) * RATE_MS / 1000.0
        ax_amp, ax_motion = axes[row]

        for a in range(3):
            ax_amp.plot(t_sec, feat[a].mean(axis=0), linewidth=1.4, label=f"antenna {a}")
        ax_amp.fill_between(t_sec, *ax_amp.get_ylim(), where=present > 0, color="tab:green", alpha=0.12,
                             step="mid", label="ground truth: present")
        ax_amp.set_ylabel(f"{activity}\namplitude (a.u.)")
        if row == 0:
            legend_handles, legend_labels = ax_amp.get_legend_handles_labels()
            ax_amp.set_title("Amplitude, subcarrier-averaged")

        ax_motion.plot(t_sec, motion, color="tab:orange", linewidth=1.4)
        ax_motion.fill_between(t_sec, 0, motion.max() * 1.05, where=present > 0, color="tab:green",
                                alpha=0.12, step="mid")
        if row == 0:
            ax_motion.set_title("Motion magnitude")

        ax_amp.set_xlabel("time (s)")
        ax_motion.set_xlabel("time (s)")

    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0),
               frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out_dir = REPO_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "features_enw_paper.png", dpi=200, bbox_inches="tight")
    plt.savefig(out_dir / "features_enw_paper.pdf", bbox_inches="tight")
    print(f"saved to {out_dir / 'features_enw_paper.png'} and .pdf")


if __name__ == "__main__":
    main()
