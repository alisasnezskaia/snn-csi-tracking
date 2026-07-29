"""Visualize the ground-truth speed_labels() targets actually used to train
and evaluate train_regression.py -- i.e. what the model is being scored
against, not the raw position trajectories (see plot_test_trajectory.py for
those).

Reuses the cached regression_rates*_meta.npz (y, groups, rates) and
re-derives each window's (activity_code, condition, rate_ms) by replaying
_discover_trials_with_gt()'s trial ordering, which is what `groups` indexes
into.

Also reproduces the train/val/test split (same per-trial chronological logic
as data.dataset.build_datasets) so the label-variance asymmetry noted in
conversation (test-set variance lower than train/val) is visible directly,
not just as a printed number.

Run:
    .venv/bin/python scripts/plot_ground_truth_labels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from snn_csi_tracking.training.train_regression import (
    RATES,
    _discover_trials_with_gt,
)

OUT_DIR = Path(__file__).parent.parent / "results" / "figures"
CACHE_DIR = Path(__file__).parent.parent / "data" / "processed"

# Points at the existing amplitude-only cache directly, not
# train_regression.cache_paths() -- that function was recently repointed to
# a new "ampphase" (amplitude+phase-diff feature) cache that hasn't been
# built yet. speed_labels() itself is unaffected by that change (labels
# don't depend on which CSI features are extracted), so this still shows
# the real, current ground-truth targets.
META_PATH = CACHE_DIR / f"regression_rates{'-'.join(str(r) for r in RATES)}_w128_s64_meta.npz"

VAL_SIZE, TEST_SIZE = 0.15, 0.15


def split_indices(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same per-trial chronological split as data.dataset.build_datasets."""
    train_idx, val_idx, test_idx = [], [], []
    for g in np.unique(groups):
        (pos,) = np.where(groups == g)
        n = len(pos)
        t1 = int(round(n * (1 - VAL_SIZE - TEST_SIZE)))
        t2 = int(round(n * (1 - TEST_SIZE)))
        train_idx.append(pos[:t1])
        val_idx.append(pos[t1:t2])
        test_idx.append(pos[t2:])
    return np.concatenate(train_idx), np.concatenate(val_idx), np.concatenate(test_idx)


def main():
    meta = np.load(META_PATH)
    y, groups, rates = meta["y"], meta["groups"], meta["rates"]
    print(f"loaded {META_PATH.name}: {len(y)} windows, {len(set(groups))} trials")

    trials = _discover_trials_with_gt()  # same order used to build `groups`
    activity_of = np.array([trials[g][3]["activity_code"] for g in groups])
    condition_of = np.array([trials[g][0] for g in groups])
    rate_ms_of = np.array([trials[g][1] for g in groups])

    train_idx, val_idx, test_idx = split_indices(groups)
    print(f"split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"label variance: train={y[train_idx].var():.2f}  val={y[val_idx].var():.2f}  "
          f"test={y[test_idx].var():.2f}  (full={y.var():.2f})")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. overall distribution, split-colored
    axes[0, 0].hist([y[train_idx], y[val_idx], y[test_idx]], bins=40, stacked=True,
                     label=["train", "val", "test"], color=["tab:blue", "tab:orange", "tab:green"])
    axes[0, 0].set_xlabel("speed_labels() value (px/frame)")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].set_title(f"label distribution by split (n={len(y)})")
    axes[0, 0].legend()

    # 2. log-scale version -- distribution is heavily right-skewed (lots of near-zero windows)
    axes[0, 1].hist(y, bins=60, color="tab:gray")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("speed_labels() value (px/frame)")
    axes[0, 1].set_ylabel("count (log scale)")
    axes[0, 1].set_title("full distribution, log y-axis")

    # 3. by activity
    codes = sorted(set(activity_of))
    axes[0, 2].boxplot([y[activity_of == c] for c in codes], tick_labels=codes, showfliers=False)
    axes[0, 2].set_ylabel("speed_labels() value (px/frame)")
    axes[0, 2].set_title("label distribution by activity")

    # 4. by capture rate
    axes[1, 0].boxplot([y[rate_ms_of == r] for r in RATES], tick_labels=[f"{r}ms" for r in RATES], showfliers=False)
    axes[1, 0].set_ylabel("speed_labels() value (px/frame)")
    axes[1, 0].set_title("label distribution by capture rate")

    # 5. example per-trial time series -- one moving, one stationary/empty
    example_trial_moving = None
    example_trial_still = None
    for g in np.unique(groups):
        code = trials[g][3]["activity_code"]
        if code == "EN-W" and example_trial_moving is None:
            example_trial_moving = g
        if code == "E" and example_trial_still is None:
            example_trial_still = g
        if example_trial_moving is not None and example_trial_still is not None:
            break
    for g, label, color in [(example_trial_moving, "EN-W (enters+walks)", "tab:red"),
                             (example_trial_still, "E (empty room)", "tab:gray")]:
        (idx,) = np.where(groups == g)
        axes[1, 1].plot(np.arange(len(idx)), y[idx], marker="o", markersize=3, label=label, color=color)
    axes[1, 1].set_xlabel("window index within trial")
    axes[1, 1].set_ylabel("speed_labels() value (px/frame)")
    axes[1, 1].set_title("example per-trial label trace")
    axes[1, 1].legend(fontsize=8)

    # 6. test-set-only distribution (what R^2 is actually computed against)
    axes[1, 2].hist(y[test_idx], bins=30, color="tab:green")
    axes[1, 2].set_xlabel("speed_labels() value (px/frame)")
    axes[1, 2].set_ylabel("count")
    axes[1, 2].set_title(f"test-split only (var={y[test_idx].var():.2f}) -- "
                          f"this is R^2's denominator")

    fig.suptitle("Ground-truth motion-magnitude labels actually used by train_regression.py")
    fig.tight_layout()
    out_path = OUT_DIR / "12_ground_truth_label_distribution.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
