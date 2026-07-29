"""Domain-adversarial (DANN) presence+position training -- directly
targets the diagnosed failure mode: leave-activity-out cross-validation
(train_presence_position_cv.py) measured presence AUROC=0.508+/-0.007
across every fold, essentially exact chance with very low variance,
confirmed independent of window length (t_win=64 and t_win=128 both landed
at the same chance level). That consistency points at the shared trunk
encoding WHICH SESSION produced a window rather than genuine occupancy.

Adds a domain classifier off the same spk2 representation the presence/
position heads use, predicting which of the fold's TRAINING sessions a
window came from, wired through a gradient-reversal layer (Ganin &
Lempitsky, 2015 -- models.presence_position_snn.grad_reverse). The domain
classifier trains normally (minimize its own cross-entropy); the trunk
feeding it gets the negated gradient, explicitly penalized for making that
classification easy -- forcing it to discard exactly the session-
identifying signal diagnosed above, while still needing to solve presence/
position from what's left.

Standard lambda ramp schedule (Ganin & Lempitsky): lambda_p =
2/(1+exp(-10*p))-1 where p = epoch/total_epochs, 0->1 over training --
starts adversarial pressure near zero so the trunk can first learn
something useful before being asked to also discard domain information.

Runs the same 4-fold leave-activity-out protocol as
train_presence_position_cv.py, --model snn only (this technique is being
validated against the diagnosed problem first; the ANN twin can get the
same treatment afterward if it helps).

Uses manual index-based batching (not the CSIDataset/DataLoader path) since
each window here also needs a domain (session) label, which the existing
Dataset classes don't carry.

Run:
    .venv/bin/python scripts/train_presence_position_dann.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_csi_tracking.data.presence_position_dataset import load_or_build_perframe_dataset
from snn_csi_tracking.models.presence_position_snn import SNNPresencePositionConvPerFrame
from snn_csi_tracking.training.train_presence_position_conv import (
    AMPLITUDE_NORM, BATCH_SIZE, CACHE_DIR, CONV_CHANNELS, DELTA_THRESHOLD, DEPTH_CACHE_DIR, HIDDEN_1,
    HIDDEN_2, KERNEL_SIZE, LEARNING_RATE, NUM_EPOCHS, RATE_MS, RAW_ROOT, SMOOTHNESS_WEIGHT, STRIDE, T_WIN,
    TRAJECTORY_CACHE_DIR, USE_EMPTY_BASELINE, USE_MOTION_MAGNITUDE, USE_PHASE, session_key,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONDITIONS = ("NLoS", "PLoS")
ROTATION = ["EN-S", "EN-W", "L", "S"]
DOMAIN_WEIGHT = 1.0


def lambda_schedule(epoch: int, total_epochs: int) -> float:
    p = epoch / total_epochs
    return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0


def fold_indices(groups, test_activity: str, val_activity: str):
    sessions = np.array([session_key(g) for g in groups])
    test_sessions = {f"{c}_{test_activity}" for c in CONDITIONS}
    val_sessions = {f"{c}_{val_activity}" for c in CONDITIONS}
    bucket = np.array(["test" if s in test_sessions else "val" if s in val_sessions else "train" for s in sessions])
    train_idx, val_idx, test_idx = np.where(bucket == "train")[0], np.where(bucket == "val")[0], np.where(bucket == "test")[0]
    train_sessions_sorted = sorted(set(sessions[train_idx]))
    domain_of = {s: i for i, s in enumerate(train_sessions_sorted)}
    train_domain = np.array([domain_of[s] for s in sessions[train_idx]])
    return train_idx, val_idx, test_idx, train_domain, len(train_sessions_sorted)


def to_batch(X, pos, present, idx, device):
    return (
        torch.as_tensor(np.asarray(X[idx]), dtype=torch.float32, device=device),
        torch.as_tensor(np.asarray(pos[idx]), dtype=torch.float32, device=device).permute(1, 0, 2),
        torch.as_tensor(np.asarray(present[idx]), dtype=torch.float32, device=device).permute(1, 0),
    )


def run_fold(X, pos, present, groups, test_activity: str, val_activity: str) -> dict:
    train_idx, val_idx, test_idx, train_domain, num_domains = fold_indices(groups, test_activity, val_activity)
    train_presence_rate = present[train_idx].mean()
    pos_weight = torch.tensor([(1 - train_presence_rate) / train_presence_rate]).to(DEVICE)

    model = SNNPresencePositionConvPerFrame(
        num_channels=X.shape[1], num_subcarriers=X.shape[2], conv_channels=CONV_CHANNELS,
        h1=HIDDEN_1, h2=HIDDEN_2, out_dim=2, kernel_size=KERNEL_SIZE, delta_threshold=DELTA_THRESHOLD,
        num_domains=num_domains,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    domain_loss_fn = nn.CrossEntropyLoss()

    best_val_loss, best_state = float("inf"), None
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        grl_lambda = lambda_schedule(epoch, NUM_EPOCHS)
        perm = np.random.permutation(len(train_idx))
        for start in range(0, len(perm), BATCH_SIZE):
            batch_pos_in_train = perm[start:start + BATCH_SIZE]
            batch_idx = train_idx[batch_pos_in_train]
            batch_domain = torch.as_tensor(train_domain[batch_pos_in_train], dtype=torch.long, device=DEVICE)
            windows, pos_seq, pres_seq = to_batch(X, pos, present, batch_idx, DEVICE)

            optimizer.zero_grad()
            pres_pred, pos_pred, domain_pred = model(windows, grl_lambda=grl_lambda)
            loss_presence = bce_loss(pres_pred, pres_seq).mean()
            pos_err = ((pos_pred - pos_seq) ** 2).sum(dim=-1)
            loss_position = (pos_err * pres_seq).sum() / (pres_seq.sum() + 1e-8)
            both_present = pres_seq[1:] * pres_seq[:-1]
            jump = ((pos_pred[1:] - pos_pred[:-1]) ** 2).sum(dim=-1)
            loss_smoothness = (jump * both_present).sum() / (both_present.sum() + 1e-8)
            # domain_pred: (T, batch, num_domains) -- same domain label at every timestep of a window
            t, b, _ = domain_pred.shape
            domain_target = batch_domain.unsqueeze(0).expand(t, b).reshape(-1)
            loss_domain = domain_loss_fn(domain_pred.reshape(t * b, -1), domain_target)
            loss = loss_presence + loss_position + SMOOTHNESS_WEIGHT * loss_smoothness + DOMAIN_WEIGHT * loss_domain
            loss.backward()
            optimizer.step()

        # validation (no domain head involved -- val sessions aren't in the domain label space)
        model.eval()
        with torch.no_grad():
            val_losses = []
            for start in range(0, len(val_idx), BATCH_SIZE):
                batch_idx = val_idx[start:start + BATCH_SIZE]
                windows, pos_seq, pres_seq = to_batch(X, pos, present, batch_idx, DEVICE)
                pres_pred, pos_pred, _domain_pred = model(windows, grl_lambda=0.0)
                loss_presence = bce_loss(pres_pred, pres_seq).mean()
                pos_err = ((pos_pred - pos_seq) ** 2).sum(dim=-1)
                mse = (pos_err * pres_seq).sum() / (pres_seq.sum() + 1e-8)
                val_losses.append((loss_presence + mse).item())
            val_loss = float(np.mean(val_losses))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_pres_pred, all_pos_pred, all_pres_gt, all_pos_gt = [], [], [], []
        for start in range(0, len(val_idx), BATCH_SIZE):
            batch_idx = val_idx[start:start + BATCH_SIZE]
            windows, pos_seq, pres_seq = to_batch(X, pos, present, batch_idx, DEVICE)
            pres_pred, _pos_pred, _domain_pred = model(windows, grl_lambda=0.0)
            all_pres_pred.append(pres_pred); all_pres_gt.append(pres_seq)
        val_pres_pred, val_pres_gt = torch.cat(all_pres_pred, dim=1), torch.cat(all_pres_gt, dim=1)
        probs = torch.sigmoid(val_pres_pred).cpu().numpy()
        gt = val_pres_gt.cpu().numpy()
        best_t, best_bacc = 0.5, -1.0
        for t_ in np.linspace(0.05, 0.95, 19):
            pred = (probs > t_).astype(float)
            pos_mask, neg_mask = gt == 1, gt == 0
            sens = pred[pos_mask].mean() if pos_mask.any() else float("nan")
            spec = (1 - pred[neg_mask]).mean() if neg_mask.any() else float("nan")
            bacc = np.nanmean([sens, spec])
            if bacc > best_bacc:
                best_bacc, best_t = bacc, float(t_)

        all_pres_pred, all_pos_pred, all_pres_gt, all_pos_gt = [], [], [], []
        for start in range(0, len(test_idx), BATCH_SIZE):
            batch_idx = test_idx[start:start + BATCH_SIZE]
            windows, pos_seq, pres_seq = to_batch(X, pos, present, batch_idx, DEVICE)
            pres_pred, pos_pred, _domain_pred = model(windows, grl_lambda=0.0)
            all_pres_pred.append(pres_pred); all_pos_pred.append(pos_pred)
            all_pres_gt.append(pres_seq); all_pos_gt.append(pos_seq)
        pres_pred, pos_pred = torch.cat(all_pres_pred, dim=1), torch.cat(all_pos_pred, dim=1)
        pres_gt, pos_gt = torch.cat(all_pres_gt, dim=1), torch.cat(all_pos_gt, dim=1)

        probs = torch.sigmoid(pres_pred).cpu().numpy().ravel()
        gt = pres_gt.cpu().numpy().ravel()
        auroc = roc_auc_score(gt, probs) if len(set(gt)) > 1 else float("nan")
        presence_acc = ((probs > best_t).astype(float) == gt).mean()
        sq_err = ((pos_pred - pos_gt) ** 2).sum(dim=-1)
        rmse = torch.sqrt((sq_err * pres_gt).sum() / (pres_gt.sum() + 1e-8)).item()

    return {"test_activity": test_activity, "presence_acc": float(presence_acc), "auroc": auroc, "rmse": rmse,
            "threshold": best_t, "num_domains": num_domains}


def main():
    print(f"Loading dataset (amplitude_norm={AMPLITUDE_NORM})...")
    X, pos, present, groups, activity_codes = load_or_build_perframe_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, DEPTH_CACHE_DIR, CACHE_DIR,
        rate_ms=RATE_MS, t_win=T_WIN, stride=STRIDE, use_3d=False, use_phase=USE_PHASE,
        use_empty_baseline=USE_EMPTY_BASELINE, denoise=None, use_motion_magnitude=USE_MOTION_MAGNITUDE,
        amplitude_norm=AMPLITUDE_NORM,
    )
    X = np.asarray(X)  # materialize once -- fold loop does many small random-index reads

    results = []
    t_start = time.time()
    for i, test_activity in enumerate(ROTATION):
        val_activity = ROTATION[(i + 1) % len(ROTATION)]
        print(f"\n=== DANN fold {i+1}/{len(ROTATION)}: test={test_activity}, val={val_activity} ===")
        t0 = time.time()
        result = run_fold(X, pos, present, groups, test_activity, val_activity)
        results.append(result)
        print(f"  presence_acc={result['presence_acc']:.3f}  auroc={result['auroc']:.3f}  "
              f"rmse={result['rmse']:.4f}  threshold={result['threshold']:.2f}  "
              f"num_domains={result['num_domains']}  ({time.time()-t0:.1f}s)")

    accs = [r["presence_acc"] for r in results]
    aurocs = [r["auroc"] for r in results if not np.isnan(r["auroc"])]
    rmses = [r["rmse"] for r in results]
    print(f"\n=== DANN cross-validation summary ({len(ROTATION)} folds) ===")
    print(f"presence_acc: {np.mean(accs):.3f} +/- {np.std(accs):.3f}   (per-fold: {[f'{a:.3f}' for a in accs]})")
    print(f"AUROC:        {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}   (per-fold: {[f'{a:.3f}' for a in aurocs]})")
    print(f"RMSE:         {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}   (per-fold: {[f'{r:.4f}' for r in rmses]})")
    print(f"total wall time: {time.time()-t_start:.1f}s")
    print(f"(compare against non-adversarial baseline: presence_acc=0.646+/-0.050, AUROC=0.508+/-0.007, RMSE=0.2606+/-0.0793)")


if __name__ == "__main__":
    main()
