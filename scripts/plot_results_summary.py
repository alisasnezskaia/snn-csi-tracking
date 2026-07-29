"""Publication-ready summary of presence-detection AUROC/accuracy across the
three evaluation regimes run tonight -- the key figure for the paper's
results/limitations section. The story it tells: presence detection works
within one continuous, uninterrupted capture file, and breaks (down to
chance) the instant a model is asked to generalize across ANY capture-file
boundary -- whether that's a completely different session (leave-activity-
out CV) or literally the next repeat within the same nominal session
(same-environment, capture-holdout). This is a sharper, more mechanistic
claim than "not enough sessions": whatever breaks generalization resets at
the capture-file boundary, not the session/day boundary (most likely AGC
re-initializing each time the CSI extraction tool starts a new file).

Hardcoded from this session's actual runs (see conversation / the
respective scripts' logs) -- not re-computed here, this is a plotting-only
script:
  - chronological (leaky, same continuous file, different time slice):
    train_presence_position_conv.py's default split-mode, presence_acc from
    the two comparable runs (z-score and energy-norm preprocessing)
  - same-environment (train captures 1-3, test capture 5, SAME session):
    train_presence_position_single_env.py, n=6 non-trivial (mixed-class) sessions
  - leave-activity-out CV (entirely different session):
    train_presence_position_cv.py, n=4 folds
  - DANN (leave-activity-out CV + domain-adversarial training):
    train_presence_position_dann.py, n=4 folds (AUROC only, 3 non-nan folds)

Run:
    .venv/bin/python scripts/plot_results_summary.py
Output: results/figures/results_summary.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent

REGIMES = [
    "Chronological\n(same continuous file,\ndifferent time slice)",
    "Same-environment\n(train captures 1-3,\ntest capture 5, SAME session)",
    "Leave-activity-out CV\n(entirely different session)",
    "DANN\n(cross-session +\ndomain-adversarial)",
]
# Chronological AUROC computed post-hoc from the saved checkpoint (single
# run, no std -- see conversation) -- the only regime with real signal.
AUROC_MEAN = [0.596, 0.515, 0.508, 0.505]
AUROC_STD = [0.0, 0.034, 0.007, 0.006]
ACC_MEAN = [0.61, 0.739, 0.646, 0.544]  # chronological: mean of the two comparable runs (0.627, 0.598)
ACC_STD = [0.02, 0.217, 0.050, 0.174]
CHANCE = 0.5


def main():
    fig, (ax_auroc, ax_acc) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(REGIMES))
    colors = ["#94a3b8", "#f59e0b", "#dc2626", "#7c3aed"]

    bars = ax_auroc.bar(x, AUROC_MEAN, yerr=AUROC_STD, capsize=5, color=colors)
    ax_auroc.axhline(CHANCE, color="black", linestyle="--", linewidth=1, label="chance (0.5)")
    ax_auroc.set_ylim(0.4, 0.65)
    ax_auroc.set_ylabel("Presence AUROC")
    ax_auroc.set_title("Presence AUROC by evaluation regime\n(threshold-independent -- the metric that reveals real generalization)")
    ax_auroc.set_xticks(x)
    ax_auroc.set_xticklabels(REGIMES, fontsize=8)
    ax_auroc.legend(fontsize=9)
    ax_auroc.grid(axis="y", alpha=0.3)

    ax_acc.bar(x, ACC_MEAN, yerr=ACC_STD, capsize=5, color=colors)
    ax_acc.axhline(CHANCE, color="black", linestyle="--", linewidth=1, label="chance-ish (0.5, class-balance dependent)")
    ax_acc.set_ylim(0, 1.0)
    ax_acc.set_ylabel("Presence accuracy (calibrated threshold)")
    ax_acc.set_title("Presence accuracy by evaluation regime\n(inflated by class imbalance -- see AUROC for the real signal)")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(REGIMES, fontsize=8)
    ax_acc.legend(fontsize=9)
    ax_acc.grid(axis="y", alpha=0.3)

    fig.suptitle("Presence detection generalizes within one continuous recording,\n"
                 "not across any capture-file boundary (same session or not)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = REPO_ROOT / "results" / "figures" / "results_summary.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
