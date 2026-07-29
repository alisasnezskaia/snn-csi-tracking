"""Position prediction, take 2 -- addresses two issues with
diagnose_position_umap.py's approach (see conversation notes):

  1. No PCA/UMAP compression before the supervised model. PCA maximizes
     retained variance, and we already showed (diagnose_position_umap.py)
     that the dominant axis of variance in this data is motion/activity
     type, not position -- so a variance-maximizing projection could easily
     be suppressing a weaker, secondary signal like position even at 50
     components. Here the full high-dimensional feature vector goes
     straight into a regularized linear model (RidgeCV) and a tree ensemble
     (RandomForest), both of which handle n << p directly without a manual
     compression step.

  2. Per-trial normalization (data.preprocessing.normalize) z-scores each
     trial by its OWN mean/std -- which, by construction, erases exactly
     the "how different is this from an empty room" signal, since an empty
     trial's own mean gets subtracted from itself too. Here, amplitude is
     instead normalized against a shared per-condition EMPTY-ROOM reference
     (mean/std pooled across that condition's "E" trials), so the feature
     for every window is literally "how far is this from what nobody-there
     looks like, in this room" -- preserving cross-trial, baseline-relative
     structure that per-trial normalization would have discarded.

Same window feature shape as diagnose_position_umap.py otherwise (per-segment
means + time std, single rate for consistent dimensionality), same grouped
(by trial) cross-validation to avoid the memorization risk from
EXPERIMENTS.md run #1.

Run:
    .venv/bin/python scripts/predict_position_baseline_relative.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from snn_csi_tracking.data import mat_loader
from snn_csi_tracking.data.preprocessing import position_labels, sliding_windows

RATE_MS = 50
WINDOW_SIZE = 128
STRIDE = 128
MAX_NAN_FRAC = 0.3
N_SEGMENTS = 8
EPS = 1e-8

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
    """windows: (n, W, Sub, Ant) -> (n, N_SEGMENTS*Sub*Ant + Sub*Ant)."""
    n, w, sub, ant = windows.shape
    segments = windows.reshape(n, N_SEGMENTS, w // N_SEGMENTS, sub, ant)
    segment_means = segments.mean(axis=2).reshape(n, -1)
    time_std = windows.std(axis=1).reshape(n, -1)
    return np.concatenate([segment_means, time_std], axis=1)


def compute_empty_room_reference(trials) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-condition (mean, std) of raw amplitude, pooled across all frames
    of all that condition's "E" trials -- the shared baseline every other
    trial gets normalized against, instead of each trial normalizing itself."""
    pooled = {}
    for condition, path, row, traj_path, is_empty in trials:
        if not is_empty:
            continue
        csi = mat_loader.load_csi(path, row["csi_key"], row["row_index"])
        if csi is None:
            continue
        amp = np.abs(csi).astype(np.float32)  # (T, Sub, Ant), raw -- no per-trial normalization
        pooled.setdefault(condition, []).append(amp)

    reference = {}
    for condition, amps in pooled.items():
        all_frames = np.concatenate(amps, axis=0)  # (sum_T, Sub, Ant)
        reference[condition] = (all_frames.mean(axis=0, keepdims=True), all_frames.std(axis=0, keepdims=True))
        print(f"  {condition} empty-room reference built from {len(amps)} trials, {all_frames.shape[0]} frames")
    return reference


def main():
    trials = discover_trials()
    n_empty = sum(1 for t in trials if t[4])
    print(f"{len(trials)} usable {RATE_MS}ms trials ({n_empty} empty-room, "
          f"{len(trials) - n_empty} occupied with <{MAX_NAN_FRAC:.0%} NaN)")

    print("\nBuilding per-condition empty-room baseline...")
    reference = compute_empty_room_reference(trials)

    features, positions_out, trial_ids, activity_codes = [], [], [], []
    trial_id = 0
    for condition, path, row, traj_path, is_empty in trials:
        if is_empty:
            continue  # position is undefined for empty-room trials -- baseline-only, not a prediction target
        mu_e, std_e = reference[condition]

        csi = mat_loader.load_csi(path, row["csi_key"], row["row_index"])
        if csi is None:
            continue
        positions = np.load(traj_path)
        if len(positions) != csi.shape[0]:
            continue

        amp_raw = np.abs(csi).astype(np.float32)
        amp = (amp_raw - mu_e) / (std_e + EPS)  # baseline-relative, NOT per-trial normalized

        windows = sliding_windows(amp, WINDOW_SIZE, STRIDE)
        labels = position_labels(positions, WINDOW_SIZE, STRIDE)

        feat = window_features(windows)
        features.append(feat)
        positions_out.append(labels)
        trial_ids.extend([trial_id] * feat.shape[0])
        activity_codes.extend([row["activity_code"]] * feat.shape[0])
        trial_id += 1
        print(f"  [{trial_id}] {condition}/{row['filename']}/{row['activity_code']}: {feat.shape[0]} windows")

    X = np.concatenate(features, axis=0)
    y = np.concatenate(positions_out, axis=0)
    trial_ids = np.array(trial_ids)
    activity_codes = np.array(activity_codes)
    print(f"\ntotal: X={X.shape}, y={y.shape}, {len(set(trial_ids))} trials")

    X_scaled = StandardScaler().fit_transform(X)  # center/scale only -- no dimensionality reduction

    n_splits = min(5, len(set(trial_ids)))
    gkf = GroupKFold(n_splits=n_splits)

    # RidgeCV: the right default tool for n << p (regularization makes the
    #   otherwise ill-posed p >> n problem well-conditioned).
    # PLSRegression: latent components chosen to covary with the TARGET,
    #   unlike PCA's target-blind variance maximization -- the principled
    #   fix for "PCA might be throwing away the position signal", not just
    #   a bigger model.
    # RandomForest: max_features="sqrt" (NOT sklearn's default of 1.0, which
    #   searches all 27k features per split -- computationally infeasible at
    #   this dimensionality, confirmed the hard way) and fewer/shallower
    #   trees, kept only as a cheap nonlinear sanity check, not the main event.
    for name, make_model in [
        ("RidgeCV", lambda: RidgeCV(alphas=np.logspace(-2, 5, 20))),
        ("PLSRegression(10 components)", lambda: PLSRegression(n_components=10)),
        ("RandomForest(sqrt features)", lambda: RandomForestRegressor(
            n_estimators=100, max_depth=6, max_features="sqrt", random_state=0, n_jobs=-1)),
    ]:
        fold_r2 = []
        for train_idx, test_idx in gkf.split(X_scaled, y, groups=trial_ids):
            model = make_model()
            model.fit(X_scaled[train_idx], y[train_idx])
            pred = np.asarray(model.predict(X_scaled[test_idx])).reshape(len(test_idx), -1)
            ss_res = ((y[test_idx] - pred) ** 2).sum()
            ss_tot = ((y[test_idx] - y[train_idx].mean(axis=0)) ** 2).sum()
            fold_r2.append(1 - ss_res / ss_tot)
        fold_r2 = np.array(fold_r2)
        print(f"\n[{name}] grouped (by trial) {n_splits}-fold R^2 (no PCA, baseline-relative features): "
              f"{fold_r2.mean():.3f} +/- {fold_r2.std():.3f}  (per fold: {np.round(fold_r2, 3)})", flush=True)

    print("\n(for comparison: diagnose_position_umap.py's PCA(50)+KNN result was R^2=0.158 "
          "on per-trial-normalized features)")


if __name__ == "__main__":
    main()
