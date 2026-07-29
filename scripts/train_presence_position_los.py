"""Leave-activity-out CV for the 3-exit architecture: presence + LoS/NLoS +
presence-gated position, all off one shared SNNPresencePositionConvPerFrame
backbone (predict_los=True -- see that class's docstring).

Motivation (see conversation): the >4x noise-floor gap measured between
NLoS_E and PLoS_E (same activity, same ground truth) is a STRUCTURED,
already-labeled confound -- the shield blocks the direct path or it
doesn't -- not unexplained per-session drift. Every leave-activity-out CV
fold run tonight (train_presence_position_cv.py) mixed both conditions
across train/val/test without ever telling the model which propagation
regime applied, asking it to learn one unified mapping across two
physically different regimes at once. This adds LoS/NLoS as an explicit
third output (not a training input -- the model has to infer it from the
signal, same as presence) so a single shared backbone can still structure
its representation around the condition, instead of either ignoring it or
training two fully separate per-condition models.

Same 4-fold leave-activity-out protocol as train_presence_position_cv.py
(E always in train, so every fold keeps real absent-class examples), same
best-of-best preprocessing (relative motion, energy amplitude
normalization). Reports presence AUROC/accuracy, position RMSE, AND
LoS/NLoS accuracy (checking the new auxiliary head's own quality) per fold,
then mean+/-std -- directly comparable to train_presence_position_cv.py's
baseline summary (presence_acc=0.646+/-0.050, AUROC=0.508+/-0.007,
RMSE=0.2606+/-0.0793) and the relative-motion-only run
(AUROC=0.542+/-0.061).

Run:
    .venv/bin/python scripts/train_presence_position_los.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_csi_tracking.data.dataset import CSIDataset
from snn_csi_tracking.data.presence_position_dataset import load_or_build_perframe_dataset
from snn_csi_tracking.models.presence_position_snn import SNNPresencePositionConvPerFrame
from snn_csi_tracking.training.train_presence_position_conv import (
    AMPLITUDE_NORM, BATCH_SIZE, CACHE_DIR, CONV_CHANNELS, DELTA_THRESHOLD, DEPTH_CACHE_DIR, HIDDEN_1,
    HIDDEN_2, KERNEL_SIZE, LEARNING_RATE, NUM_EPOCHS, NUM_WORKERS, RATE_MS, RAW_ROOT, SMOOTHNESS_WEIGHT,
    STRIDE, T_WIN, TRAJECTORY_CACHE_DIR, USE_EMPTY_BASELINE, USE_MOTION_MAGNITUDE, USE_PHASE,
    USE_RELATIVE_MOTION, session_key,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONDITIONS = ("NLoS", "PLoS")
ROTATION = ["EN-S", "EN-W", "L", "S"]  # E excluded -- always kept in train
LOS_WEIGHT = 1.0


def condition_labels(groups: np.ndarray) -> np.ndarray:
    """1.0 for PLoS, 0.0 for NLoS -- which condition maps to which value is
    arbitrary, just needs to be consistent between labels and reported acc."""
    return np.array([1.0 if g.startswith("PLoS") else 0.0 for g in groups], dtype=np.float32)


def make_fold_loaders(X, pos, present, condition, groups, test_activity: str, val_activity: str):
    sessions = np.array([session_key(g) for g in groups])
    test_sessions = {f"{c}_{test_activity}" for c in CONDITIONS}
    val_sessions = {f"{c}_{val_activity}" for c in CONDITIONS}
    bucket = np.array([
        "test" if s in test_sessions else "val" if s in val_sessions else "train" for s in sessions
    ])
    train_idx, val_idx, test_idx = np.where(bucket == "train")[0], np.where(bucket == "val")[0], np.where(bucket == "test")[0]
    train_ds = CSIDataset(X, pos, train_idx, label_dtype=torch.float32, extra=present, extra2=condition)
    val_ds = CSIDataset(X, pos, val_idx, label_dtype=torch.float32, extra=present, extra2=condition)
    test_ds = CSIDataset(X, pos, test_idx, label_dtype=torch.float32, extra=present, extra2=condition)
    loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), persistent_workers=NUM_WORKERS > 0)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs),
        train_ds,
    )


def train_one_epoch_los(model, loader, optimizer, bce_loss) -> float:
    model.train()
    total_loss, total = 0.0, 0
    for windows, pos_seq, pres_seq, cond in loader:
        windows = windows.to(DEVICE)
        pos_seq = pos_seq.to(DEVICE).permute(1, 0, 2)
        pres_seq = pres_seq.to(DEVICE).permute(1, 0)
        cond = cond.to(DEVICE)  # (batch,) -- constant per window, not per-timestep
        optimizer.zero_grad()
        pres_pred, pos_pred, los_pred = model(windows)
        loss_presence = bce_loss(pres_pred, pres_seq).mean()
        pos_err = ((pos_pred - pos_seq) ** 2).sum(dim=-1)
        loss_position = (pos_err * pres_seq).sum() / (pres_seq.sum() + 1e-8)
        both_present = pres_seq[1:] * pres_seq[:-1]
        jump = ((pos_pred[1:] - pos_pred[:-1]) ** 2).sum(dim=-1)
        loss_smoothness = (jump * both_present).sum() / (both_present.sum() + 1e-8)
        cond_seq = cond.unsqueeze(0).expand_as(los_pred)  # same condition label at every timestep
        loss_los = nn.functional.binary_cross_entropy_with_logits(los_pred, cond_seq)
        loss = loss_presence + loss_position + SMOOTHNESS_WEIGHT * loss_smoothness + LOS_WEIGHT * loss_los
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pres_seq.shape[1]
        total += pres_seq.shape[1]
    return total_loss / total


@torch.no_grad()
def collect_predictions_los(model, loader):
    model.eval()
    all_pres_pred, all_pos_pred, all_los_pred, all_pres, all_pos, all_cond = [], [], [], [], [], []
    for windows, pos_seq, pres_seq, cond in loader:
        windows = windows.to(DEVICE)
        pos_seq = pos_seq.to(DEVICE).permute(1, 0, 2)
        pres_seq = pres_seq.to(DEVICE).permute(1, 0)
        cond = cond.to(DEVICE)
        pres_pred, pos_pred, los_pred = model(windows)
        all_pres_pred.append(pres_pred); all_pos_pred.append(pos_pred); all_los_pred.append(los_pred)
        all_pres.append(pres_seq); all_pos.append(pos_seq)
        all_cond.append(cond.unsqueeze(0).expand_as(los_pred))
    return (torch.cat(all_pres_pred, dim=1), torch.cat(all_pos_pred, dim=1), torch.cat(all_los_pred, dim=1),
            torch.cat(all_pres, dim=1), torch.cat(all_pos, dim=1), torch.cat(all_cond, dim=1))


def presence_auroc(pres_pred: torch.Tensor, pres_gt: torch.Tensor) -> float:
    probs = torch.sigmoid(pres_pred).cpu().numpy().ravel()
    gt = pres_gt.cpu().numpy().ravel()
    return roc_auc_score(gt, probs) if len(set(gt)) > 1 else float("nan")


def calibrate_threshold(pred: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    probs = torch.sigmoid(pred).cpu().numpy().ravel()
    g = gt.cpu().numpy().ravel()
    best_t, best_bacc = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        p = (probs > t).astype(float)
        pos_mask, neg_mask = g == 1, g == 0
        sens = p[pos_mask].mean() if pos_mask.any() else float("nan")
        spec = (1 - p[neg_mask]).mean() if neg_mask.any() else float("nan")
        bacc = np.nanmean([sens, spec])
        if bacc > best_bacc:
            best_bacc, best_t = bacc, float(t)
    return best_t, best_bacc


def run_fold(X, pos, present, condition, groups, test_activity: str, val_activity: str) -> dict:
    train_loader, val_loader, test_loader, train_ds = make_fold_loaders(
        X, pos, present, condition, groups, test_activity, val_activity
    )
    train_presence_rate = train_ds.extra.float().mean().item()
    pos_weight = torch.tensor([(1 - train_presence_rate) / train_presence_rate]).to(DEVICE)

    model = SNNPresencePositionConvPerFrame(
        num_channels=X.shape[1], num_subcarriers=X.shape[2], conv_channels=CONV_CHANNELS,
        h1=HIDDEN_1, h2=HIDDEN_2, out_dim=2, kernel_size=KERNEL_SIZE, delta_threshold=DELTA_THRESHOLD,
        predict_los=True,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    best_val_loss, best_state = float("inf"), None
    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch_los(model, train_loader, optimizer, bce_loss)
        pres_pred, pos_pred, _los_pred, pres_gt, pos_gt, _cond_gt = collect_predictions_los(model, val_loader)
        loss_presence = bce_loss(pres_pred, pres_gt).mean()
        sq_err = ((pos_pred - pos_gt) ** 2).sum(dim=-1)
        mse = (sq_err * pres_gt).sum() / (pres_gt.sum() + 1e-8)
        val_loss = (loss_presence + mse).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    val_pres_pred, _val_pos_pred, val_los_pred, val_pres_gt, _val_pos_gt, val_cond_gt = collect_predictions_los(model, val_loader)
    presence_threshold, _ = calibrate_threshold(val_pres_pred, val_pres_gt)
    los_threshold, _ = calibrate_threshold(val_los_pred, val_cond_gt)

    pres_pred, pos_pred, los_pred, pres_gt, pos_gt, cond_gt = collect_predictions_los(model, test_loader)
    presence_probs = torch.sigmoid(pres_pred)
    presence_acc = ((presence_probs > presence_threshold).float() == pres_gt).float().mean().item()
    auroc = presence_auroc(pres_pred, pres_gt)
    sq_err = ((pos_pred - pos_gt) ** 2).sum(dim=-1)
    rmse = torch.sqrt((sq_err * pres_gt).sum() / (pres_gt.sum() + 1e-8)).item()
    los_probs = torch.sigmoid(los_pred)
    los_acc = ((los_probs > los_threshold).float() == cond_gt).float().mean().item()

    return {"test_activity": test_activity, "presence_acc": presence_acc, "auroc": auroc, "rmse": rmse,
            "los_acc": los_acc, "presence_threshold": presence_threshold, "los_threshold": los_threshold}


def main():
    print(f"Loading dataset (amplitude_norm={AMPLITUDE_NORM}, use_relative_motion={USE_RELATIVE_MOTION})...")
    X, pos, present, groups, activity_codes = load_or_build_perframe_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, DEPTH_CACHE_DIR, CACHE_DIR,
        rate_ms=RATE_MS, t_win=T_WIN, stride=STRIDE, use_3d=False, use_phase=USE_PHASE,
        use_empty_baseline=USE_EMPTY_BASELINE, denoise=None, use_motion_magnitude=USE_MOTION_MAGNITUDE,
        amplitude_norm=AMPLITUDE_NORM, use_relative_motion=USE_RELATIVE_MOTION,
    )
    condition = condition_labels(groups)

    results = []
    t_start = time.time()
    for i, test_activity in enumerate(ROTATION):
        val_activity = ROTATION[(i + 1) % len(ROTATION)]
        print(f"\n=== LoS-aware fold {i+1}/{len(ROTATION)}: test={test_activity}, val={val_activity} ===")
        t0 = time.time()
        result = run_fold(X, pos, present, condition, groups, test_activity, val_activity)
        results.append(result)
        print(f"  presence_acc={result['presence_acc']:.3f}  auroc={result['auroc']:.3f}  "
              f"rmse={result['rmse']:.4f}  los_acc={result['los_acc']:.3f}  ({time.time()-t0:.1f}s)")

    accs = [r["presence_acc"] for r in results]
    aurocs = [r["auroc"] for r in results if not np.isnan(r["auroc"])]
    rmses = [r["rmse"] for r in results]
    los_accs = [r["los_acc"] for r in results]
    print(f"\n=== LoS-aware 3-exit cross-validation summary ({len(ROTATION)} folds) ===")
    print(f"presence_acc: {np.mean(accs):.3f} +/- {np.std(accs):.3f}   (per-fold: {[f'{a:.3f}' for a in accs]})")
    print(f"AUROC:        {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}   (per-fold: {[f'{a:.3f}' for a in aurocs]})")
    print(f"RMSE:         {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}   (per-fold: {[f'{r:.4f}' for r in rmses]})")
    print(f"LoS/NLoS acc: {np.mean(los_accs):.3f} +/- {np.std(los_accs):.3f}   (per-fold: {[f'{a:.3f}' for a in los_accs]})")
    print(f"total wall time: {time.time()-t_start:.1f}s")
    print("(compare against relative-motion baseline: presence_acc=0.627+/-0.066, AUROC=0.542+/-0.061, RMSE=0.3043+/-0.0792)")


if __name__ == "__main__":
    main()
