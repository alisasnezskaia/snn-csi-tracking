"""Standalone LoS/NLoS (shield present or not) classifier -- a completely
separate, much simpler model from the presence+position SNN, built after
discovering (see conversation) that trying to recover this condition as an
auxiliary output of the dynamic, delta-encoded SNN only reached 0.588+/-
0.104 accuracy (barely above chance on a binary task), because that
pipeline is built entirely around DYNAMIC, motion-based features, while
LoS/NLoS is a STATIC property of the whole recording (the shield blocks
the direct path or it doesn't, for the entire capture).

The fix: use a static feature instead -- each capture's own time-AVERAGED
raw amplitude spectrum (one number per antenna/subcarrier, no motion
information at all) -- with a plain logistic regression. This got 100%
accuracy and AUROC on every leave-activity-out fold tested.

Given that reliability, this is meant to run as a genuinely separate,
cheap PRE-PROCESSING stage: classify the condition once per capture/
deployment, then feed that (~100% trustworthy) label into the main
presence+position model as a known input (see models/presence_position_snn.
py's condition_embed_dim / scripts/train_presence_position_condition_input.py)
-- not as something the main dynamic model has to infer itself.

Run:
    .venv/bin/python scripts/train_los_classifier.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_csi_tracking.data.raw_capture_loader import discover_captures, parse_csi_file
from snn_csi_tracking.training.train_presence_position_conv import RAW_ROOT

ACTIVITIES = ["E", "EN-S", "EN-W", "L", "S"]


def build_fingerprints() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, sessions): X is (NumCaptures, NumAntennas*NumSubcarriers)
    float32 -- each capture's time-averaged |CSI| spectrum, a STATIC
    fingerprint with no motion information. y is 1 for PLoS, 0 for NLoS.
    sessions is "{condition}_{activity}", for leave-activity-out grouping."""
    captures = discover_captures(RAW_ROOT)
    fingerprints, conditions, sessions = [], [], []
    for cap in captures:
        csi, _timestamps = parse_csi_file(cap.csi_path)
        mean_spectrum = np.abs(csi).mean(axis=2).ravel()  # (Ant*Sub,) -- averaged over time
        fingerprints.append(mean_spectrum)
        conditions.append(1 if cap.condition == "PLoS" else 0)
        sessions.append(f"{cap.condition}_{cap.activity}")
    return np.stack(fingerprints).astype(np.float32), np.array(conditions), np.array(sessions)


def main():
    print("Building per-capture static amplitude fingerprints...")
    X, y, sessions = build_fingerprints()
    print(f"fingerprint shape: {X.shape}, class balance: {y.mean():.3f}")

    accs, aurocs = [], []
    for held_out in ACTIVITIES:
        test_mask = np.array([s.endswith(held_out) for s in sessions])
        train_mask = ~test_mask
        scaler = StandardScaler().fit(X[train_mask])
        x_train, x_test = scaler.transform(X[train_mask]), scaler.transform(X[test_mask])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(x_train, y[train_mask])
        probs = clf.predict_proba(x_test)[:, 1]
        preds = clf.predict(x_test)
        acc = (preds == y[test_mask]).mean()
        auroc = roc_auc_score(y[test_mask], probs)
        accs.append(acc)
        aurocs.append(auroc)
        print(f"held-out activity={held_out:5s}: acc={acc:.3f}  auroc={auroc:.3f}  n_test={test_mask.sum()}")

    print(f"\nmean acc={np.mean(accs):.3f}+/-{np.std(accs):.3f}  mean auroc={np.mean(aurocs):.3f}+/-{np.std(aurocs):.3f}")


if __name__ == "__main__":
    main()
