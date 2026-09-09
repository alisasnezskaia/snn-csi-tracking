"""Leave-activity-out cross-validation for the joint presence+position task.

With only 10 sessions, a single fixed train/val/test split can land on an
unrepresentative test session -- train_presence_position_conv.py's
BALANCED_SPLIT, for example, puts NLoS_L in the test set and scores
AUROC=0.419 there, worse than chance, which no threshold-based
postprocessing can fix. Averaging metrics across folds distinguishes an
unlucky split from a genuine generalization problem, and is the
methodology expected at this sample size.

4 folds, one per non-empty-room activity (EN-S, EN-W, L, S) held out as
test in turn -- E is never held out, so every fold's training set always
includes real absent-class examples (see BALANCED_SPLIT's own docstring on
why that invariant matters). Each fold:
    test  = {NLoS_<activity>, PLoS_<activity>}          (2 sessions)
    val   = {NLoS_<next>, PLoS_<next>}                   (2 sessions, a
             different held-out-rotation activity, for checkpoint selection
             + presence-threshold calibration -- never the test activity)
    train = the remaining 6 sessions (always includes E)

Reports presence_acc, presence AUROC (threshold-independent -- surfaces
failures raw accuracy would hide), and position RMSE per fold, then mean
+/- std across folds -- for both --model snn and --model ann, same config
as train_presence_position_conv.py otherwise.

Run:
    .venv/bin/python scripts/train_presence_position_cv.py --model snn
    .venv/bin/python scripts/train_presence_position_cv.py --model ann
"""

from __future__ import annotations

import argparse
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
from snn_csi_tracking.data.raw_capture_loader import NUM_ANTENNAS
from snn_csi_tracking.models.presence_position_ann import ANNPresencePositionConvPerFrame
from snn_csi_tracking.models.presence_position_lstm import LSTMPresencePositionConvPerFrame
from snn_csi_tracking.models.presence_position_snn import SNNPresencePositionConvPerFrame
from snn_csi_tracking.training.train_presence_position_conv import (
    AMPLITUDE_NORM, BATCH_SIZE, CACHE_DIR, CONV_CHANNELS, DELTA_THRESHOLD, DEPTH_CACHE_DIR, HIDDEN_1,
    HIDDEN_2, KERNEL_SIZE, LEARNING_RATE, NUM_EPOCHS, NUM_WORKERS, RATE_MS, RAW_ROOT, SMOOTHNESS_WEIGHT,
    STRIDE, T_WIN, TRAJECTORY_CACHE_DIR, USE_EMPTY_BASELINE, USE_MOTION_MAGNITUDE, USE_PHASE, USE_RELATIVE_MOTION,
    calibrate_presence_threshold, collect_predictions, evaluate, identity_norm_stats, position_norm_stats, presence_balanced_acc,
    rmse_and_presence_acc, session_key, train_one_epoch,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONDITIONS = ("NLoS", "PLoS")
ROTATION = ["EN-S", "EN-W", "L", "S"]  # E excluded -- always kept in train, see module docstring


def make_fold_loaders(X, pos, present, groups, test_activity: str, val_activity: str, batch_size: int):
    sessions = np.array([session_key(g) for g in groups])
    test_sessions = {f"{c}_{test_activity}" for c in CONDITIONS}
    val_sessions = {f"{c}_{val_activity}" for c in CONDITIONS}
    bucket = np.array([
        "test" if s in test_sessions else "val" if s in val_sessions else "train" for s in sessions
    ])
    train_idx, val_idx, test_idx = np.where(bucket == "train")[0], np.where(bucket == "val")[0], np.where(bucket == "test")[0]
    train_ds = CSIDataset(X, pos, train_idx, label_dtype=torch.float32, extra=present)
    val_ds = CSIDataset(X, pos, val_idx, label_dtype=torch.float32, extra=present)
    test_ds = CSIDataset(X, pos, test_idx, label_dtype=torch.float32, extra=present)
    loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), persistent_workers=NUM_WORKERS > 0)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        train_ds,
    )


def presence_auroc(pres_pred: torch.Tensor, pres_gt: torch.Tensor) -> float:
    probs = torch.sigmoid(pres_pred).cpu().numpy().ravel()
    gt = pres_gt.cpu().numpy().ravel()
    if len(set(gt)) < 2:
        return float("nan")  # single-class session: AUROC is undefined
    return roc_auc_score(gt, probs)


def run_fold(model_type: str, X, pos, present, groups, test_activity: str, val_activity: str,
             encoder_type: str = "perframe", num_static_channels: int = 0, out_dim: int = 2,
             pinhole: bool = True) -> dict:
    train_loader, val_loader, test_loader, train_ds = make_fold_loaders(
        X, pos, present, groups, test_activity, val_activity, BATCH_SIZE
    )
    train_presence_rate = train_ds.extra.float().mean().item()
    pos_weight = torch.tensor([(1 - train_presence_rate) / train_presence_rate]).to(DEVICE)
    # position_norm_stats only earns its keep for real-meters pinhole positions (a genuine
    # BCE-vs-position scale mismatch to fix); 2D and 3D no-pinhole are already [0,1] by
    # construction, where a data-driven per-fold rescale adds risk (see identity_norm_stats).
    mu, sigma = position_norm_stats(train_ds) if (out_dim == 3 and pinhole) else identity_norm_stats(out_dim)

    if model_type == "snn":
        model = SNNPresencePositionConvPerFrame(
            num_channels=X.shape[1], num_subcarriers=X.shape[2], conv_channels=CONV_CHANNELS,
            h1=HIDDEN_1, h2=HIDDEN_2, out_dim=out_dim, kernel_size=KERNEL_SIZE, delta_threshold=DELTA_THRESHOLD,
            encoder_type=encoder_type, num_static_channels=num_static_channels,
        ).to(DEVICE)
    elif model_type == "ann":
        model = ANNPresencePositionConvPerFrame(
            num_channels=X.shape[1], num_subcarriers=X.shape[2], conv_channels=CONV_CHANNELS,
            h1=HIDDEN_1, h2=HIDDEN_2, out_dim=out_dim, kernel_size=KERNEL_SIZE,
        ).to(DEVICE)
    else:
        # lstm -- same conv frontend/layer sizes as the ANN twin (see
        # models/presence_position_lstm.py). No encoder_type/num_static_channels
        # (those are SNN-only options). Uses the same calibrated-threshold
        # presence_acc metric as SNN/ANN, rather than uncalibrated balanced
        # accuracy.
        model = LSTMPresencePositionConvPerFrame(
            num_channels=X.shape[1], num_subcarriers=X.shape[2], conv_channels=CONV_CHANNELS,
            h1=HIDDEN_1, h2=HIDDEN_2, out_dim=out_dim, kernel_size=KERNEL_SIZE,
        ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    # Checkpoint selection by best val BALANCED accuracy (fixed threshold=0.5),
    # per explicit request: presence accuracy is the metric that matters here,
    # not AUROC. Balanced (not raw) accuracy specifically to avoid the one
    # documented failure mode in this codebase -- a degenerate "always
    # predict present" epoch scoring artificially high raw accuracy just
    # because a given fold's validation split happens to be imbalanced (see
    # calibrate_presence_threshold's docstring). The final reported number
    # is still the ordinary (raw, calibrated-threshold) presence_acc below --
    # this only changes which of the NUM_EPOCHS checkpoints gets kept.
    best_val_bacc, best_state = -1.0, None
    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, bce_loss, SMOOTHNESS_WEIGHT, mu, sigma)
        val_loss, val_presence_acc, val_rmse = evaluate(model, val_loader, bce_loss, mu, sigma)
        val_pres_pred, _val_pos_pred, val_pres_gt, _val_pos_gt = collect_predictions(model, val_loader)
        val_bacc = presence_balanced_acc(val_pres_pred, val_pres_gt, threshold=0.5)
        print(f"    epoch {epoch}/{NUM_EPOCHS}  val_loss={val_loss:.4f}  "
              f"val_presence_acc={val_presence_acc:.3f}  val_bacc={val_bacc:.3f}  val_rmse={val_rmse:.4f}")
        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is None:  # every val epoch was single-class (val_bacc all NaN) -- fall back to the last epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    calib_threshold, _calib_bacc = calibrate_presence_threshold(model, val_loader)
    test_pres_pred, test_pos_pred, test_pres_gt, test_pos_gt = collect_predictions(model, test_loader)
    rmse, presence_acc = rmse_and_presence_acc(test_pres_pred, test_pos_pred, test_pres_gt, test_pos_gt,
                                                threshold=calib_threshold, mu=mu, sigma=sigma)
    auroc = presence_auroc(test_pres_pred, test_pres_gt)
    return {"test_activity": test_activity, "val_activity": val_activity, "calib_threshold": calib_threshold,
            "presence_acc": presence_acc, "auroc": auroc, "rmse": rmse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["snn", "ann", "lstm"], required=True)
    parser.add_argument("--rate-ms", type=int, default=RATE_MS, help="capture rate: 5, 10, 50, or 100")
    parser.add_argument("--t-win", type=int, default=T_WIN, help="window length in frames")
    parser.add_argument("--stride", type=int, default=STRIDE, help="window stride in frames")
    parser.add_argument("--feature", choices=["none", "relative_motion", "spectral_ratio", "cross_coherence",
                                               "spectral_and_coherence", "both", "full", "relmot_and_coherence"],
                         default="relative_motion",
                         help="which extra motion-derived channel(s) to use -- none (bare amplitude+phase, no "
                              "extra channel -- the true zero-feature-engineering baseline), relative_motion "
                              "(rolling-ratio), spectral_ratio (low/high frequency power ratio), cross_coherence "
                              "(magnitude-squared coherence between antenna pairs -- does structure AGREE across "
                              "independent sensors), spectral_and_coherence (both new features together), "
                              "both (relative_motion + spectral_ratio), full (the complete Eq. (9) feature stack "
                              "-- raw motion magnitude + relative motion + spectral ratio + cross-coherence "
                              "together; implies --use-motion-magnitude, no need to pass it separately), "
                              "relmot_and_coherence (relative_motion + cross_coherence together, nothing else -- "
                              "AP+CC+MM's counterpart when normalized motion, not raw, wins the AP+MM vs "
                              "AP+tilde_m comparison; note AP+CC+MM itself needs no new choice here, it's just "
                              "--feature cross_coherence --use-motion-magnitude, since raw MM is independent)")
    parser.add_argument("--encoder", choices=["perframe", "timeaware"], default="perframe",
                         help="perframe: PerFrameConvEncoder (1D conv, one frame at a time, the original design). "
                              "timeaware: TimeAwareConvEncoder (real 2D conv across subcarrier AND several "
                              "timesteps jointly -- best standalone result of the investigation, AUROC=0.650, "
                              "tested here integrated into the real dual-head presence+position architecture)")
    parser.add_argument("--baseline-deviation", action="store_true",
                         help="add the empty-room-baseline-deviation channel (see presence_position_dataset."
                              "empty_baseline_deviation_feature) -- a STATIC presence signal (this trial's amplitude "
                              "vs the condition's person-free reference), unlike every --feature option above which "
                              "is motion-based. Targets the presence-flicker-during-sitting failure mode observed "
                              "on the L activity (person sits still, ground truth=present, motion features go quiet)")
    parser.add_argument("--use-motion-magnitude", action="store_true",
                         help="add the raw (absolute) frame-to-frame motion-magnitude channel (see "
                              "presence_position_dataset.motion_magnitude_feature) -- ORTHOGONAL to --feature "
                              "(which never touches this flag), so the two compose for a cumulative ablation: "
                              "e.g. --feature none (amp+phase only) -> --feature none --use-motion-magnitude "
                              "(+motion magnitude) -> --feature spectral_ratio --use-motion-magnitude "
                              "(+spectral ratio) -> --feature spectral_and_coherence --use-motion-magnitude "
                              "(+cross-coherence), a 4-stage cumulative feature ablation")
    parser.add_argument("--use-3d", action="store_true",
                         help="real camera-frame (X,Y,Z) in meters instead of normalized image-plane (x,y) -- "
                              "see camera_geometry.deproject_pixel_to_camera_frame")
    parser.add_argument("--no-pinhole", dest="pinhole", action="store_false", default=True,
                         help="only meaningful with --use-3d: skip the pinhole deprojection and keep (x, y) as "
                              "normalized [0,1] image-plane values with z = metric depth min-max normalized to "
                              "[0,1] over the whole dataset -- RMSE stays comparable in scale to the 2D-only run")
    args = parser.parse_args()
    out_dim = 3 if args.use_3d else 2
    use_relative_motion = args.feature in ("relative_motion", "both", "full", "relmot_and_coherence")
    use_spectral_ratio = args.feature in ("spectral_ratio", "both", "spectral_and_coherence", "full")
    use_cross_coherence = args.feature in ("cross_coherence", "spectral_and_coherence", "full", "relmot_and_coherence")
    use_motion_magnitude = args.use_motion_magnitude or args.feature == "full"

    print(f"Loading dataset (amplitude_norm={AMPLITUDE_NORM}, rate_ms={args.rate_ms}, "
          f"t_win={args.t_win}, stride={args.stride}, window_duration={args.t_win * args.rate_ms / 1000:.2f}s, "
          f"feature={args.feature}, use_motion_magnitude={use_motion_magnitude}, "
          f"use_3d={args.use_3d}, pinhole={args.pinhole})...")
    X, pos, present, groups, activity_codes = load_or_build_perframe_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, DEPTH_CACHE_DIR, CACHE_DIR,
        rate_ms=args.rate_ms, t_win=args.t_win, stride=args.stride, use_3d=args.use_3d, pinhole=args.pinhole,
        use_phase=USE_PHASE,
        use_empty_baseline=USE_EMPTY_BASELINE, denoise=None, use_motion_magnitude=use_motion_magnitude,
        amplitude_norm=AMPLITUDE_NORM, use_relative_motion=use_relative_motion, use_spectral_ratio=use_spectral_ratio,
        use_cross_coherence=use_cross_coherence, use_baseline_deviation=args.baseline_deviation,
    )

    results = []
    t_start = time.time()
    for i, test_activity in enumerate(ROTATION):
        val_activity = ROTATION[(i + 1) % len(ROTATION)]
        print(f"\n=== fold {i+1}/{len(ROTATION)}: test={test_activity}, val={val_activity}, model={args.model} ===")
        t0 = time.time()
        result = run_fold(args.model, X, pos, present, groups, test_activity, val_activity, encoder_type=args.encoder,
                           num_static_channels=NUM_ANTENNAS if args.baseline_deviation else 0, out_dim=out_dim,
                           pinhole=args.pinhole)
        results.append(result)
        print(f"  presence_acc={result['presence_acc']:.3f}  auroc={result['auroc']:.3f}  "
              f"rmse={result['rmse']:.4f}  threshold={result['calib_threshold']:.2f}  ({time.time()-t0:.1f}s)")

    accs = [r["presence_acc"] for r in results]
    aurocs = [r["auroc"] for r in results if not np.isnan(r["auroc"])]
    rmses = [r["rmse"] for r in results]
    print(f"\n=== {args.model.upper()} cross-validation summary ({len(ROTATION)} folds) ===")
    print(f"presence_acc: {np.mean(accs):.3f} +/- {np.std(accs):.3f}   (per-fold: {[f'{a:.3f}' for a in accs]})")
    print(f"AUROC:        {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}   (per-fold: {[f'{a:.3f}' for a in aurocs]})")
    print(f"RMSE:         {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}   (per-fold: {[f'{r:.4f}' for r in rmses]})")
    print(f"total wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
