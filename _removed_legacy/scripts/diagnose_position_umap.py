"""Unsupervised diagnostic: does CSI amplitude carry any position-correlated
structure at all, in this fixed room -- before spending a training run on it?

No model training here. Per (non-overlapping) window: reduce the raw CSI
amplitude to a feature vector, standardize, PCA down to a manageable
dimensionality, then UMAP to 2D for visualization.

v2 -- two fixes after v1 showed no position structure (see conversation
notes), to rule out "the analysis method is wrong" before concluding
"the signal isn't there":

  1. Empty-room ("E") trials are now included, not filtered out. v1 only
     compared already-occupied windows against each other, so it never
     actually tested the much bigger, well-established presence/absence
     effect (a person disrupts many multipaths at once). If E doesn't
     separate cleanly from occupied trials, that would mean the pipeline
     itself is broken, not just that position is hard -- this is the
     sanity check for that.
  2. The per-window feature is no longer a single flat mean-over-time
     (which throws away exactly the kind of temporal-change information
     that motivates delta encoding in the actual model, see encoding.py).
     It's now N_SEGMENTS sub-window means (coarse temporal shape) plus the
     per-(subcarrier,antenna) std over the full window (how much it
     varied), concatenated.

Single rate only (50ms) -- 5/10/50ms trials have a 3-antenna x 1024-subcarrier
CSI shape, but the 100ms trials only have 1 antenna x 64 subcarriers (see
mat_loader.py), so mixing rates would confound "different feature dimension"
with "different position". 50ms chosen as a reasonable window-count/
trial-count balance among the 3-antenna rates.

Two outputs:
  1. results/figures/11_umap_position_diagnostic.png -- 2D UMAP embedding,
     colored by presence (E vs. occupied), ground-truth x, ground-truth y,
     trial id (memorization check), and activity.
  2. A grouped (by trial) cross-validated KNN-regression R^2, predicting
     (x, y) from the pre-UMAP PCA features, restricted to occupied windows
     (position is undefined/degenerate for empty-room windows) -- GroupKFold
     so no window from a test trial's own group leaks into training.

Run:
    .venv/bin/python scripts/diagnose_position_umap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import umap

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from snn_csi_tracking.data import mat_loader
from snn_csi_tracking.data.preprocessing import normalize, position_labels, sliding_windows

RATE_MS = 50
WINDOW_SIZE = 128
STRIDE = 128  # non-overlapping: avoids near-duplicate windows biasing the
              # embedding/CV toward trial fingerprints (see EXPERIMENTS.md run #1)
MAX_NAN_FRAC = 0.3  # exclude occupied trials where MediaPipe mostly failed to
                     # detect anyone (see conversation notes on ~20 problem
                     # trials) -- not applied to "E" trials, which are
                     # expected to be ~100% NaN by construction (nobody there)
N_SEGMENTS = 8  # window split into this many sub-segments for the temporal-
                # shape feature (128 frames / 8 = 16 frames per segment)

DATA_ROOT = Path(__file__).parent.parent / "data" / "raw"
TRAJECTORY_DIR = Path(__file__).parent.parent / "data" / "processed" / "trajectories"
OUT_DIR = Path(__file__).parent.parent / "results" / "figures"


def discover_trials() -> list[tuple[str, Path, dict, Path, bool]]:
    trials = []
    for condition in ("NLoS", "PLoS"):
        path = DATA_ROOT / condition / f"csi_office_{RATE_MS}ms_interframe.mat"
        if not path.exists():
            continue
        for row in mat_loader.load_table_metadata(path):
            code = row["activity_code"]
            is_empty = code == "E"
            traj_path = TRAJECTORY_DIR / f"{condition}_{RATE_MS}ms_{code}_{row['filename']}.npy"
            if not traj_path.exists():
                continue
            if not is_empty:
                positions = np.load(traj_path)
                if np.isnan(positions[:, 0]).mean() > MAX_NAN_FRAC:
                    continue
            trials.append((condition, path, row, traj_path, is_empty))
    return trials


def window_features(windows: np.ndarray) -> np.ndarray:
    """windows: (n, W, Sub, Ant). Returns (n, N_SEGMENTS*Sub*Ant + Sub*Ant):
    per-segment means (coarse temporal shape) concatenated with the
    full-window std (how much each (subcarrier, antenna) varied) -- unlike a
    flat time-mean, this keeps some of the temporal-change information the
    actual model's delta encoding relies on."""
    n, w, sub, ant = windows.shape
    segments = windows.reshape(n, N_SEGMENTS, w // N_SEGMENTS, sub, ant)
    segment_means = segments.mean(axis=2).reshape(n, -1)  # (n, N_SEGMENTS*Sub*Ant)
    time_std = windows.std(axis=1).reshape(n, -1)  # (n, Sub*Ant)
    return np.concatenate([segment_means, time_std], axis=1)


def main():
    trials = discover_trials()
    n_empty = sum(1 for t in trials if t[4])
    print(f"{len(trials)} usable {RATE_MS}ms trials ({n_empty} empty-room, "
          f"{len(trials) - n_empty} occupied with <{MAX_NAN_FRAC:.0%} NaN)")

    features, positions_out, trial_ids, activity_codes, is_empty_flags = [], [], [], [], []
    for trial_id, (condition, path, row, traj_path, is_empty) in enumerate(trials):
        csi = mat_loader.load_csi(path, row["csi_key"], row["row_index"])
        if csi is None:
            continue
        positions = np.load(traj_path)
        if len(positions) != csi.shape[0]:
            continue

        amp = normalize(csi)  # (T, NumSubcarriers, NumAntennas), per-trial z-scored
        windows = sliding_windows(amp, WINDOW_SIZE, STRIDE)  # (n, W, Sub, Ant)
        labels = position_labels(positions, WINDOW_SIZE, STRIDE)  # (n, 2) -- degenerate (0,0) if is_empty

        feat = window_features(windows)
        features.append(feat)
        positions_out.append(labels)
        trial_ids.extend([trial_id] * feat.shape[0])
        activity_codes.extend([row["activity_code"]] * feat.shape[0])
        is_empty_flags.extend([is_empty] * feat.shape[0])
        print(f"  [{trial_id+1}/{len(trials)}] {condition}/{row['filename']}/{row['activity_code']}: "
              f"{feat.shape[0]} windows")

    X = np.concatenate(features, axis=0)
    y = np.concatenate(positions_out, axis=0)
    trial_ids = np.array(trial_ids)
    activity_codes = np.array(activity_codes)
    is_empty_flags = np.array(is_empty_flags)
    print(f"\ntotal: X={X.shape}, y={y.shape}, {len(set(trial_ids))} trials, "
          f"{is_empty_flags.sum()} empty-room windows")

    # --- quantitative check 1: grouped-CV KNN regression on PCA features, occupied windows only ---
    X_scaled = StandardScaler().fit_transform(X)
    n_pca = min(50, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_pca, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA({n_pca}) explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    occ = ~is_empty_flags
    n_splits = min(5, len(set(trial_ids[occ])))
    gkf = GroupKFold(n_splits=n_splits)
    fold_r2 = []
    for train_idx, test_idx in gkf.split(X_pca[occ], y[occ], groups=trial_ids[occ]):
        knn = KNeighborsRegressor(n_neighbors=10)
        knn.fit(X_pca[occ][train_idx], y[occ][train_idx])
        pred = knn.predict(X_pca[occ][test_idx])
        ss_res = ((y[occ][test_idx] - pred) ** 2).sum()
        ss_tot = ((y[occ][test_idx] - y[occ][train_idx].mean(axis=0)) ** 2).sum()
        fold_r2.append(1 - ss_res / ss_tot)
    fold_r2 = np.array(fold_r2)
    print(f"\n[position] grouped (by trial) {n_splits}-fold KNN R^2 on PCA features: "
          f"{fold_r2.mean():.3f} +/- {fold_r2.std():.3f}  (per fold: {np.round(fold_r2, 3)})")

    # --- quantitative check 2: grouped-CV KNN classification, presence (E vs occupied) ---
    from sklearn.neighbors import KNeighborsClassifier

    n_splits_p = min(5, len(set(trial_ids)))
    gkf_p = GroupKFold(n_splits=n_splits_p)
    presence_acc = []
    for train_idx, test_idx in gkf_p.split(X_pca, is_empty_flags, groups=trial_ids):
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(X_pca[train_idx], is_empty_flags[train_idx])
        presence_acc.append(knn.score(X_pca[test_idx], is_empty_flags[test_idx]))
    presence_acc = np.array(presence_acc)
    baseline_acc = max(is_empty_flags.mean(), 1 - is_empty_flags.mean())
    print(f"[presence] grouped {n_splits_p}-fold KNN accuracy (E vs occupied): "
          f"{presence_acc.mean():.3f} +/- {presence_acc.std():.3f}  "
          f"(majority-class baseline: {baseline_acc:.3f})")

    # --- visualization ---
    embedding = umap.UMAP(n_components=2, random_state=0).fit_transform(X_pca)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].scatter(embedding[occ, 0], embedding[occ, 1], s=6, c="tab:blue", label="occupied", alpha=0.6)
    axes[0, 0].scatter(embedding[~occ, 0], embedding[~occ, 1], s=6, c="tab:red", label="empty room", alpha=0.6)
    axes[0, 0].set_title(f"presence (KNN acc={presence_acc.mean():.2f}, baseline={baseline_acc:.2f})")
    axes[0, 0].legend(fontsize=7)

    sc0 = axes[0, 1].scatter(embedding[occ, 0], embedding[occ, 1], c=y[occ, 0], cmap="viridis", s=6)
    axes[0, 1].set_title("colored by ground-truth x (occupied only)")
    fig.colorbar(sc0, ax=axes[0, 1])

    sc1 = axes[0, 2].scatter(embedding[occ, 0], embedding[occ, 1], c=y[occ, 1], cmap="viridis", s=6)
    axes[0, 2].set_title("colored by ground-truth y (occupied only)")
    fig.colorbar(sc1, ax=axes[0, 2])

    axes[1, 0].scatter(embedding[:, 0], embedding[:, 1], c=trial_ids, cmap="tab20", s=6)
    axes[1, 0].set_title("colored by trial id (memorization check)")

    for code in sorted(set(activity_codes)):
        mask = activity_codes == code
        axes[1, 1].scatter(embedding[mask, 0], embedding[mask, 1], s=6, label=code, alpha=0.6)
    axes[1, 1].set_title("colored by activity")
    axes[1, 1].legend(fontsize=7)

    axes[1, 2].axis("off")

    fig.suptitle(f"UMAP of {RATE_MS}ms CSI window features (v2: segments+std) -- "
                 f"position R^2={fold_r2.mean():.3f}, presence acc={presence_acc.mean():.2f}")
    fig.tight_layout()
    out_path = OUT_DIR / "11_umap_position_diagnostic.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
