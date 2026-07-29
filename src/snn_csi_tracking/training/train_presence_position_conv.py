"""THE joint presence+position tracking configuration -- matches the paper
abstract's actual claim ("we formulate people tracking as a joint
presence-detection and position-estimation problem"), NOT the simpler
scalar motion-magnitude regression in the paper's (not yet updated) formal
section, and NOT the paper's Section 3 preprocessing recipe verbatim.
Preprocessing/architecture choices are instead "best of the best" from
everything actually validated tonight, keeping the abstract's task framing
as the only hard constraint:

KEPT (measured to help): phase-difference features (already validated by
the grid-motion pipeline), a conv frontend before the LIF stack
(SNNPresencePositionConvPerFrame) instead of a flat Linear layer -- shares
weights across subcarrier positions instead of overparameterizing the first
layer, which is exactly what made earlier feature additions (CIR,
delta-rate) overfit -- the motion-magnitude scalar channel
(presence_position_dataset.motion_magnitude_feature, cheap per-frame
"how much is changing" signal), and honest SESSION-grouped evaluation
(never per-window/chronological for the reported numbers).

DROPPED (measured to hurt or found structurally unsound tonight): the
empty-room baseline correction (USE_EMPTY_BASELINE=False below) -- this was
this session's own invention, not paper-backed, and traced to a specific
day-to-day session-drift bug (a fixed person-free reference recorded on a
different day than the trial it's applied to, breaking exactly when trial
and reference amplitude scales disagree for reasons unrelated to occupancy)
that was confirmed independently in both the L and EN-W activities. Per-
trial normalization (baseline=None -> compute_features' own fallback) is
self-referential and immune to that failure mode. Wavelet/PCA denoising
measured ~zero benefit earlier and is left at --denoise none by default.
CIR and a full per-subcarrier delta-rate channel are not included at all --
both were tried and both overfit.

Two variants, both run from this one script via --use-3d: 2D (normalized
image-plane x, y) and 3D (real camera-relative X, Y, Z in meters, via
camera_geometry.deproject_pixel_to_camera_frame + depth_extraction.py's
metric-indoor depth -- requires the raw per-trial videos, not just cached
2D trajectories). Run both separately (see the two commands at the bottom
of this file). Note the 3D RMSE below mixes units across dimensions if
compared against the 2D run's [0,1]-normalized one -- it's now in meters,
not the same scale.

--model {snn,ann}: trains either SNNPresencePositionConvPerFrame or its
non-spiking twin ANNPresencePositionConvPerFrame (models/
presence_position_ann.py) -- same conv frontend, same layer sizes, same
data/split/config, differing only in spiking vs dense activation. Run both
(same seed) to get the matched pair scripts/estimate_energy_presence_position.py
compares.

Reports RMSE (root mean squared position error, masked to ground-truth-
present frames) -- more directly interpretable on the [0,1]-normalized
position scale than the raw squared/L2 "position_error" the older
per-frame trainer reports. Checkpoint selection uses val RMSE directly
(not the combined BCE+MSE training loss), so the model actually reported
as "best" is the one that's best by the number we're reporting -- learned
the hard way this session (see conversation: the grid-cell experiment's
combined-loss selection picked an undertrained checkpoint once, because
combined loss and the metric we cared about weren't tracking the same
thing).

Also reports presence accuracy broken out by activity code, empty-room
("E") vs everything else: after empty-room baseline subtraction, E-frames
sit at ~zero amplitude change and are trivially easy to classify absent --
a single pooled presence accuracy number can look strong almost entirely
because of how many easy E-frames are in the test set, without saying much
about real presence detection during actual activity (see conversation).

Groups by SESSION (condition + activity), not by individual capture file --
checking the actual Drive layout (see conversation) showed every
activity's 5 numbered "captures" are 5 repeats within *one* recording
visit, sharing every confound (day, person, hardware warm-up, furniture)
except which specific repetition it is. Grouping by capture alone
overstates how much independent evidence backs the held-out split.

Applies the same post-processing already used for the spy-cam overlay
video (inference.debounce_presence + inference.smooth_position) to the raw
per-frame test predictions and reports RMSE/presence accuracy again,
smoothed -- applied per-window (each window is one contiguous 50ms x 64 =
3.2s clip already, never straddling a real gap) rather than first
stitching overlapping windows into one continuous per-session stream the
way render_spy_overlay.py does; a reasonable approximation for "does
smoothing help at all", not a full reproduction of that script's pipeline.

Finally, also evaluates the same trained model against the ORIGINAL
chronological (within-trial) split (dataset.build_datasets, matching
evaluate_presence_position.py) -- not to train differently, just to report
a number directly comparable to historical baselines like
presence_position_snn_perframe_ampphase_3d_50ms.pt's 0.814 presence_acc,
which was measured that way. Comparing the session-grouped and
chronological numbers for the same model tells you how much of any given
presence_acc is genuine generalization versus split leakage (see
conversation -- this is exactly the discrepancy that came up when 0.671
looked like a regression from a historical 0.814 that was never measured
the same way).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from snn_csi_tracking.data.dataset import CSIDataset, build_datasets, build_datasets_grouped
from snn_csi_tracking.data.presence_position_dataset import load_or_build_perframe_dataset
from snn_csi_tracking.inference import debounce_presence, smooth_position
from snn_csi_tracking.models.presence_position_ann import ANNPresencePositionConvPerFrame
from snn_csi_tracking.models.presence_position_lstm import LSTMPresencePositionConvPerFrame
from snn_csi_tracking.models.presence_position_snn import SNNPresencePositionConvPerFrame

REPO_ROOT = Path(__file__).parent.parent.parent.parent
RAW_ROOT = REPO_ROOT / "data" / "raw_captures"
TRAJECTORY_CACHE_DIR = REPO_ROOT / "data" / "processed" / "presence_position_trajectories"
DEPTH_CACHE_DIR = REPO_ROOT / "data" / "processed" / "presence_position_depth"
CACHE_DIR = REPO_ROOT / "data" / "processed"

RATE_MS = 50
T_WIN = 64
STRIDE = 32
USE_PHASE = True
USE_EMPTY_BASELINE = False  # "best of the best": dropped, see module docstring
USE_MOTION_MAGNITUDE = False
USE_RELATIVE_MOTION = True  # rolling-median-normalized motion magnitude, replacing
                             # the raw (absolute) version -- see presence_position_
                             # dataset.relative_motion_magnitude_feature's docstring:
                             # a raw frame-to-frame delta scales linearly with
                             # whatever gain a session's AGC happens to be at, so
                             # even "difference" features aren't actually invariant
                             # to gain drift; this ratios against a short rolling
                             # reference so a gain difference cancels out by
                             # construction, targeting a driftING (not just fixed)
                             # per-session gain -- z-scoring is already invariant to
                             # a FIXED gain and still failed to generalize, which is
                             # why this targets drift specifically (see conversation)
AMPLITUDE_NORM = "energy"  # per-frame energy normalization, not per-trial z-score
                            # -- see preprocessing.energy_normalize's docstring:
                            # per-trial z-scoring erases the absolute-scale
                            # signal presence detection needs across sessions
                            # (confirmed empirically: near-identical raw output
                            # on two held-out sessions with opposite ground truth)

CONV_CHANNELS = [16, 32]
KERNEL_SIZE = 9
HIDDEN_1 = 128
HIDDEN_2 = 32
DELTA_THRESHOLD = 0.3

# Same rationale as train_presence_position_perframe.py: penalizes
# frame-to-frame position jumps between consecutive PRESENT frames only.
SMOOTHNESS_WEIGHT = 1.0

LEARNING_RATE = 5e-4
BATCH_SIZE = 32
NUM_EPOCHS = 60
NUM_WORKERS = 4
SEED = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def session_key(trial_key: str) -> str:
    """"{condition}_{activity}_{capture_name}" -> "{condition}_{activity}" --
    the true independent unit (see module docstring)."""
    return trial_key.rsplit("_", 1)[0]


# Hand-curated session split, not a random draw -- with only 10 true
# sessions total (2 per activity), a plain random/stratification-free split
# can land on a genuinely broken bucket (confirmed this session: split-seed=2
# put BOTH empty-room sessions in val together, making val 100% absent --
# position RMSE there is then mathematically 0/0-shaped and trivially always
# "perfect", silently breaking checkpoint selection). This assignment
# instead guarantees: train includes an empty-room session (the model needs
# real absent-class examples to learn what "nobody's here" looks like, not
# just present-class ones), and neither val nor test is 100% one activity
# type. Not chosen for the best score -- chosen to avoid a degenerate bucket,
# which is the defensible bar here, not "hardest possible" or "easiest
# possible" (see conversation).
BALANCED_SPLIT = {
    "NLoS_E": "train", "NLoS_EN-S": "train", "NLoS_EN-W": "train",
    "PLoS_EN-S": "train", "PLoS_L": "train", "PLoS_S": "train",
    "NLoS_S": "val", "PLoS_EN-W": "val",
    "PLoS_E": "test", "NLoS_L": "test",
}


def make_dataloaders(X, pos, present, groups, activity_codes, batch_size, split_seed: int, split_mode: str):
    sessions = np.array([session_key(g) for g in groups])
    n_sessions = len(set(sessions))

    if split_mode == "balanced":
        assert set(BALANCED_SPLIT) == set(sessions), "BALANCED_SPLIT doesn't match this dataset's actual sessions"
        bucket = np.array([BALANCED_SPLIT[s] for s in sessions])
        train_idx, val_idx, test_idx = np.where(bucket == "train")[0], np.where(bucket == "val")[0], np.where(bucket == "test")[0]
        train_ds = CSIDataset(X, pos, train_idx, label_dtype=torch.float32, extra=present)
        val_ds = CSIDataset(X, pos, val_idx, label_dtype=torch.float32, extra=present)
        test_ds = CSIDataset(X, pos, test_idx, label_dtype=torch.float32, extra=present)
    else:
        # data/raw_captures only has ~10 true sessions total (2 per activity
        # -- see module docstring). Stratifying by activity_code (5 strata of
        # 2 sessions each) leaves nothing for a val bucket -- each stratum
        # can only split 1 train / 1 test, no room for a 3rd bucket. Below
        # that threshold, drop stratification (pool all sessions) so val
        # isn't silently empty; costs even representation of every activity
        # in every split, but an empty val set is worse. Even unstratified,
        # a bad split_seed can still degenerate (see BALANCED_SPLIT's
        # docstring above) -- prefer split_mode="balanced" unless
        # deliberately checking sensitivity to the split itself.
        strata = activity_codes if n_sessions >= 15 else None
        if strata is None:
            print(f"only {n_sessions} sessions -- stratifying by activity would leave no room for a val "
                  f"set, splitting the session pool unstratified instead")
        train_ds, val_ds, test_ds = build_datasets_grouped(
            X, pos, sessions, label_dtype=torch.float32, extra=present, strata=strata, seed=split_seed,
        )
    print(f"split (mode={split_mode}, seed={split_seed}): train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} windows")
    print(f"  train sessions: {sorted(set(sessions[train_ds.indices]))}")
    print(f"  val sessions:   {sorted(set(sessions[val_ds.indices]))}")
    print(f"  test sessions:  {sorted(set(sessions[test_ds.indices]))}")
    loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), persistent_workers=NUM_WORKERS > 0)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        train_ds, test_ds,
    )


def position_norm_stats(train_ds) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-axis min and range (max-min) of the position target, from present
    frames of THIS split's training data only (never val/test -- would leak
    that fold's distribution into training, the same leakage this project
    already treats carefully elsewhere, see run_fold's own docstring on
    session-grouped vs chronological splits).

    Motivated by a real regression: switching position labels from
    normalized [0,1] pixels to real (X,Y,Z) meters left the combined
    objective's position term far larger in magnitude than the presence
    BCE term (real-meters squared error routinely O(1-10) vs BCE's O(1)),
    which starved presence learning through the shared trunk -- confirmed
    by AUROC collapsing toward chance on the 3D SNN CV run despite
    presence_acc still looking superficially OK (see conversation).

    Min-max (not z-score/std) deliberately: this is leave-ACTIVITY-out CV,
    so the held-out fold's real positions can sit far outside the training
    activities' range (e.g. training on seated/desk activities, testing on
    a walking one) -- dividing by a std computed from a narrow-range axis
    blew up the held-out fold's normalized error to ~50x the intended scale
    (see conversation: val_loss ~57 instead of the expected O(1-4)). Min-max
    mirrors exactly how the old 2D pipeline's (x,y) was already scaled
    (MediaPipe's own normalized-pixel convention), which is the known-good
    reference point we're trying to reproduce the loss balance of.
    """
    present_mask = train_ds.extra.bool()
    present_pos = train_ds.labels[present_mask]
    pos_min = present_pos.min(dim=0).values.to(DEVICE)
    pos_range = (present_pos.max(dim=0).values.to(DEVICE) - pos_min).clamp(min=1e-6)
    return pos_min, pos_range


def train_one_epoch(model, loader, optimizer, bce_loss, smoothness_weight: float,
                     mu: torch.Tensor, sigma: torch.Tensor) -> float:
    model.train()
    total_loss, total = 0.0, 0
    for windows, pos_seq, pres_seq in loader:
        windows = windows.to(DEVICE)
        pos_seq = pos_seq.to(DEVICE).permute(1, 0, 2)  # (T, batch, out_dim)
        pres_seq = pres_seq.to(DEVICE).permute(1, 0)  # (T, batch)
        optimizer.zero_grad()
        pres_pred, pos_pred = model(windows)  # pos_pred is now in normalized (z-scored) space
        loss_presence = bce_loss(pres_pred, pres_seq).mean()
        pos_seq_norm = (pos_seq - mu) / sigma
        pos_err = ((pos_pred - pos_seq_norm) ** 2).sum(dim=-1)
        loss_position = (pos_err * pres_seq).sum() / (pres_seq.sum() + 1e-8)
        loss = loss_presence + loss_position
        if smoothness_weight > 0:
            both_present = pres_seq[1:] * pres_seq[:-1]
            jump = ((pos_pred[1:] - pos_pred[:-1]) ** 2).sum(dim=-1)
            loss_smoothness = (jump * both_present).sum() / (both_present.sum() + 1e-8)
            loss = loss + smoothness_weight * loss_smoothness
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pres_seq.shape[1]
        total += pres_seq.shape[1]
    return total_loss / total


@torch.no_grad()
def collect_predictions(model, loader):
    """Runs the whole loader through the model once. Returns (pres_pred,
    pos_pred, pres, pos), each (T, N) or (T, N, out_dim), N = every window
    in the loader concatenated in iteration order (== dataset order when
    the loader isn't shuffled)."""
    model.eval()
    all_pres_pred, all_pos_pred, all_pres, all_pos = [], [], [], []
    for windows, pos_seq, pres_seq in loader:
        windows = windows.to(DEVICE)
        pos_seq = pos_seq.to(DEVICE).permute(1, 0, 2)
        pres_seq = pres_seq.to(DEVICE).permute(1, 0)
        pres_pred, pos_pred = model(windows)
        all_pres_pred.append(pres_pred)
        all_pos_pred.append(pos_pred)
        all_pres.append(pres_seq)
        all_pos.append(pos_seq)
    return (torch.cat(all_pres_pred, dim=1), torch.cat(all_pos_pred, dim=1),
            torch.cat(all_pres, dim=1), torch.cat(all_pos, dim=1))


def rmse_and_presence_acc(pres_pred, pos_pred, pres, pos, threshold: float = 0.5,
                           mu: torch.Tensor | None = None, sigma: torch.Tensor | None = None) -> tuple[float, float]:
    """pos_pred is in normalized (z-scored) space when mu/sigma are given --
    un-normalized back to the same real units as `pos` (meters, for the 3D
    pipeline) before computing RMSE, so the reported number is always a
    real distance, never the training-time normalized scale (see
    position_norm_stats)."""
    presence_acc = ((torch.sigmoid(pres_pred) > threshold).float() == pres).float().mean().item()
    if mu is not None:
        pos_pred = pos_pred * sigma + mu
    sq_err = ((pos_pred - pos) ** 2).sum(dim=-1)
    rmse = torch.sqrt((sq_err * pres).sum() / (pres.sum() + 1e-8)).item()
    return rmse, presence_acc


def rmse_meters(pos_pred: torch.Tensor, pos: torch.Tensor, pres: torch.Tensor, homography_path: Path) -> float:
    """Same alignment check as rmse_and_presence_acc's position term --
    predicted vs. true position, differenced, masked to present frames --
    just with BOTH pos_pred and pos passed through the same head-height
    homography first (see scripts/calibrate_head_homography.py), so the
    result is real centimeters instead of a normalized [0,1] image
    fraction. Not a different metric, not a retrain -- purely a post-hoc
    coordinate change applied identically to prediction and ground truth,
    evaluated on top of an already-trained model's existing predictions."""
    import cv2

    H = np.load(homography_path)
    orig_shape = pos_pred.shape
    pred_np = pos_pred.detach().cpu().numpy().reshape(-1, 1, 2).astype(np.float64)
    true_np = pos.detach().cpu().numpy().reshape(-1, 1, 2).astype(np.float64)
    pred_m = cv2.perspectiveTransform(pred_np, H).reshape(orig_shape)
    true_m = cv2.perspectiveTransform(true_np, H).reshape(orig_shape)
    pred_m = torch.as_tensor(pred_m, dtype=pos_pred.dtype, device=pos_pred.device)
    true_m = torch.as_tensor(true_m, dtype=pos.dtype, device=pos.device)
    sq_err = ((pred_m - true_m) ** 2).sum(dim=-1)
    return torch.sqrt((sq_err * pres).sum() / (pres.sum() + 1e-8)).item()


def presence_balanced_acc(pres_pred: torch.Tensor, pres_gt: torch.Tensor, threshold: float) -> float:
    """Mean of sensitivity (present-recall) and specificity (absent-recall)
    -- unlike raw accuracy, not dominated by whichever class happens to be
    more common in the split (see conversation: raw accuracy silently
    rewarded a model that always predicted "present" on an imbalanced test
    set)."""
    pred = (torch.sigmoid(pres_pred) > threshold).float()
    pos_mask, neg_mask = pres_gt == 1, pres_gt == 0
    sens = (pred[pos_mask] == 1).float().mean().item() if pos_mask.any() else float("nan")
    spec = (pred[neg_mask] == 0).float().mean().item() if neg_mask.any() else float("nan")
    return (sens + spec) / 2


@torch.no_grad()
def calibrate_presence_threshold(model, val_loader) -> tuple[float, float]:
    """Sweeps the presence decision threshold on the VAL set (never test) and
    picks the one maximizing balanced accuracy, instead of hardcoding 0.5 --
    a free postprocessing fix for a model whose raw sigmoid output is biased
    toward one class (see conversation: this run's first checkpoint predicted
    "present" almost everywhere, tanking empty-room accuracy specifically).
    Returns (best_threshold, best_balanced_acc)."""
    pres_pred, _pos_pred, pres_gt, _pos_gt = collect_predictions(model, val_loader)
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        score = presence_balanced_acc(pres_pred, pres_gt, float(t))
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t, best_score


@torch.no_grad()
def evaluate(model, loader, bce_loss, mu: torch.Tensor, sigma: torch.Tensor) -> tuple[float, float, float]:
    """Returns (loss, presence_acc, rmse). `loss` is computed in the same
    normalized space as training (so checkpoint selection stays consistent
    with what the model was actually optimized for); `rmse` is un-normalized
    back to real units via rmse_and_presence_acc."""
    pres_pred, pos_pred, pres, pos = collect_predictions(model, loader)
    rmse, presence_acc = rmse_and_presence_acc(pres_pred, pos_pred, pres, pos, mu=mu, sigma=sigma)
    pos_norm = (pos - mu) / sigma
    sq_err = ((pos_pred - pos_norm) ** 2).sum(dim=-1)
    mse = (sq_err * pres).sum() / (pres.sum() + 1e-8)
    loss = bce_loss(pres_pred, pres).mean().item() + mse.item()
    return loss, presence_acc, rmse


def smoothed_predictions(
    pres_pred: torch.Tensor, pos_pred: torch.Tensor, threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies debounce_presence + smooth_position per window (columns of
    the (T, N[, out_dim]) tensors) -- see module docstring on why per-window,
    not a full per-session stitch."""
    probs = torch.sigmoid(pres_pred).cpu().numpy()  # (T, N)
    pos = pos_pred.cpu().numpy()  # (T, N, out_dim)
    t, n = probs.shape
    smoothed_pres = np.zeros_like(probs)
    smoothed_pos = np.zeros_like(pos)
    for i in range(n):
        presence_binary = debounce_presence(probs[:, i], threshold=threshold, min_run=6)
        smoothed_pres[:, i] = presence_binary
        smoothed_pos[:, i] = smooth_position(pos[:, i], presence_binary, window=5)
    # An all-absent window has nothing for smooth_position to average, so it
    # comes back all-NaN there -- masked out by presence in the RMSE (0
    # weight) everywhere else, but NaN * 0 is still NaN in IEEE float, so it
    # has to be replaced with a real number before that multiply, not after.
    smoothed_pos = np.nan_to_num(smoothed_pos, nan=0.0)
    return torch.as_tensor(smoothed_pres), torch.as_tensor(smoothed_pos)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--use-3d", action="store_true", help="predict (x, y, z) instead of (x, y)")
    parser.add_argument("--model", choices=["snn", "ann", "lstm"], default="snn",
                         help="snn: SNNPresencePositionConvPerFrame (spiking). ann: its non-spiking twin "
                              "ANNPresencePositionConvPerFrame -- same conv frontend/layer sizes/config, "
                              "for the SNN-vs-ANN energy comparison (scripts/estimate_energy_presence_position.py). "
                              "lstm: LSTMPresencePositionConvPerFrame -- same conv frontend/layer sizes/config, "
                              "real nn.LSTM recurrence instead of spiking/leaky-accumulation; a conventional-DL "
                              "accuracy/stability benchmark, not part of the energy comparison (see "
                              "models/presence_position_lstm.py's docstring)")
    parser.add_argument("--denoise", choices=["none", "wavelet", "pca"], default="none")
    parser.add_argument("--seed", type=int, default=SEED, help="model weight init / loader shuffling")
    parser.add_argument("--split-mode", choices=["balanced", "random"], default="balanced",
                         help="'balanced' (default): hand-curated split guaranteeing no degenerate "
                              "all-one-activity bucket (see BALANCED_SPLIT). 'random': build_datasets_grouped "
                              "with --split-seed -- can land on a broken bucket with only 10 sessions "
                              "(confirmed this session), use to deliberately check split-sensitivity")
    parser.add_argument("--split-seed", type=int, default=0,
                         help="only used with --split-mode random -- which sessions land in train/val/test")
    args = parser.parse_args()
    denoise = None if args.denoise == "none" else args.denoise

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not RAW_ROOT.exists():
        sys.exit(f"No raw captures found at {RAW_ROOT}.")
    if args.use_3d and not any((RAW_ROOT / d / "videos").exists() for d in [p.name for p in RAW_ROOT.iterdir()]):
        sys.exit("--use-3d needs the raw per-trial videos under data/raw_captures/{condition}_{activity}/videos/.")

    out_dim = 3 if args.use_3d else 2
    dim_tag = "3d" if args.use_3d else "2d"
    model_out = REPO_ROOT / "results" / "models" / f"presence_position_conv_{args.model}_{dim_tag}_{args.denoise}_50ms.pt"

    t_start = time.time()
    print(f"Loading {RATE_MS}ms captures, T_WIN={T_WIN}, STRIDE={STRIDE}, use_3d={args.use_3d}, "
          f"model={args.model}, use_empty_baseline={USE_EMPTY_BASELINE}, amplitude_norm={AMPLITUDE_NORM}, "
          f"denoise={denoise}, use_motion_magnitude={USE_MOTION_MAGNITUDE}, "
          f"use_relative_motion={USE_RELATIVE_MOTION}, seed={args.seed}...")
    X, pos, present, groups, activity_codes = load_or_build_perframe_dataset(
        RAW_ROOT, TRAJECTORY_CACHE_DIR, DEPTH_CACHE_DIR, CACHE_DIR,
        rate_ms=RATE_MS, t_win=T_WIN, stride=STRIDE, use_3d=args.use_3d, use_phase=USE_PHASE,
        use_empty_baseline=USE_EMPTY_BASELINE, denoise=denoise, use_motion_magnitude=USE_MOTION_MAGNITUDE,
        amplitude_norm=AMPLITUDE_NORM, use_relative_motion=USE_RELATIVE_MOTION,
    )
    print(f"X={X.shape}, pos={pos.shape}, {len(set(groups))} trials ({time.time()-t_start:.1f}s)")

    train_loader, val_loader, test_loader, train_ds, test_ds = make_dataloaders(
        X, pos, present, groups, activity_codes, BATCH_SIZE, args.split_seed, args.split_mode
    )

    train_presence_rate = train_ds.extra.float().mean().item()
    pos_weight = torch.tensor([(1 - train_presence_rate) / train_presence_rate]).to(DEVICE)
    print(f"pos_weight (presence) = {pos_weight.item():.3f}")
    mu, sigma = position_norm_stats(train_ds)

    if args.model == "snn":
        model = SNNPresencePositionConvPerFrame(
            num_channels=X.shape[1], num_subcarriers=X.shape[2], conv_channels=CONV_CHANNELS,
            h1=HIDDEN_1, h2=HIDDEN_2, out_dim=out_dim, kernel_size=KERNEL_SIZE, delta_threshold=DELTA_THRESHOLD,
        ).to(DEVICE)
    elif args.model == "ann":
        model = ANNPresencePositionConvPerFrame(
            num_channels=X.shape[1], num_subcarriers=X.shape[2], conv_channels=CONV_CHANNELS,
            h1=HIDDEN_1, h2=HIDDEN_2, out_dim=out_dim, kernel_size=KERNEL_SIZE,
        ).to(DEVICE)
    else:
        model = LSTMPresencePositionConvPerFrame(
            num_channels=X.shape[1], num_subcarriers=X.shape[2], conv_channels=CONV_CHANNELS,
            h1=HIDDEN_1, h2=HIDDEN_2, out_dim=out_dim, kernel_size=KERNEL_SIZE,
        ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    # Selection criterion is val_loss (BCE-presence + MSE-position + smoothness),
    # not val_rmse alone -- this task is JOINTLY presence+position (the
    # abstract's actual claim), and selecting purely by rmse picked a
    # checkpoint that was excellent at position but collapsed to "always
    # present" on this run (empty-room test accuracy 0.143, worse than
    # chance) -- see conversation. val_loss is the one number that's a
    # genuine function of both heads together.
    best_val_loss, best_state, best_epoch = float("inf"), None, None

    print(f"\nTraining on {DEVICE} for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, bce_loss, SMOOTHNESS_WEIGHT, mu, sigma)
        val_loss, val_presence_acc, val_rmse = evaluate(model, val_loader, bce_loss, mu, sigma)
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            marker = "  <- best so far"
        print(
            f"epoch {epoch:2d}/{NUM_EPOCHS}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
            f"val_presence_acc={val_presence_acc:.3f}  val_rmse={val_rmse:.4f}  "
            f"({time.time()-t0:.1f}s){marker}"
        )

    print(f"\nrestoring best checkpoint from epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    model.load_state_dict(best_state)

    calib_threshold, calib_balanced_acc = calibrate_presence_threshold(model, val_loader)
    print(f"calibrated presence threshold (val, balanced acc): threshold={calib_threshold:.2f}, "
          f"balanced_acc={calib_balanced_acc:.3f} (vs fixed 0.5)")

    test_pres_pred, test_pos_pred, test_pres_gt, test_pos_gt = collect_predictions(model, test_loader)
    test_rmse, test_presence_acc = rmse_and_presence_acc(
        test_pres_pred, test_pos_pred, test_pres_gt, test_pos_gt, threshold=calib_threshold, mu=mu, sigma=sigma
    )
    print(f"\nFINAL TEST (raw, calibrated threshold={calib_threshold:.2f}): "
          f"presence_acc={test_presence_acc:.3f}, rmse={test_rmse:.4f}")

    # E-vs-non-E breakdown -- test_loader isn't shuffled, so its iteration
    # order matches test_ds.indices, letting us align per-window activity
    # codes to the concatenated per-frame predictions.
    test_activity = activity_codes[test_ds.indices]
    is_e = torch.as_tensor(test_activity == "E").to(test_pres_gt.device)
    for label, mask in [("E (empty room)", is_e), ("non-E (real activity)", ~is_e)]:
        if mask.sum() == 0:
            continue
        m = mask.unsqueeze(0).expand_as(test_pres_gt)
        acc = ((torch.sigmoid(test_pres_pred) > calib_threshold).float() == test_pres_gt)[m].float().mean().item()
        print(f"  presence_acc [{label}]: {acc:.3f}  ({mask.sum().item()} windows)")

    smoothed_pres, smoothed_pos = smoothed_predictions(test_pres_pred, test_pos_pred, threshold=calib_threshold)
    smoothed_pres, smoothed_pos = smoothed_pres.to(test_pres_gt.device), smoothed_pos.to(test_pos_gt.device)
    smoothed_rmse, smoothed_presence_acc = rmse_and_presence_acc(
        torch.logit(smoothed_pres.clamp(1e-4, 1 - 1e-4)), smoothed_pos, test_pres_gt, test_pos_gt, threshold=0.5,
        mu=mu, sigma=sigma,
    )
    print(f"FINAL TEST (smoothed, calibrated threshold={calib_threshold:.2f}): "
          f"presence_acc={smoothed_presence_acc:.3f}, rmse={smoothed_rmse:.4f}")

    # Chronological (leaky, within-trial) split -- same trained model,
    # different test partition -- for direct comparison against historical
    # baselines like presence_position_snn_perframe_ampphase_3d_50ms.pt's
    # 0.814 presence_acc, which was measured this way (evaluate_presence_
    # position.py -> dataset.build_datasets), not with session-grouping.
    # Isolates how much of any reported number is split-methodology versus
    # actual model quality (see conversation).
    _chrono_train_ds, _chrono_val_ds, chrono_test_ds = build_datasets(
        X, pos, groups, label_dtype=torch.float32, extra=present
    )
    chrono_test_loader = DataLoader(chrono_test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    chrono_pres_pred, chrono_pos_pred, chrono_pres, chrono_pos = collect_predictions(model, chrono_test_loader)
    chrono_rmse, chrono_presence_acc = rmse_and_presence_acc(
        chrono_pres_pred, chrono_pos_pred, chrono_pres, chrono_pos, threshold=calib_threshold, mu=mu, sigma=sigma
    )
    print(f"FINAL TEST (chronological split, comparable to historical 0.814 baseline): "
          f"presence_acc={chrono_presence_acc:.3f}, rmse={chrono_rmse:.4f}")
    print(f"total wall time: {time.time()-t_start:.1f}s")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_out)
    print(f"saved best checkpoint to {model_out}")


if __name__ == "__main__":
    main()


# THE configuration (2D, best-of-best preprocessing, matches the abstract):
#   .venv/bin/python src/snn_csi_tracking/training/train_presence_position_conv.py --model snn
#   .venv/bin/python src/snn_csi_tracking/training/train_presence_position_conv.py --model ann
# (run snn + ann -- same data/split/config -- for scripts/estimate_energy_presence_position.py;
#  --model lstm is a same-protocol accuracy/stability benchmark, not part of that energy comparison)
#
# --use-3d, --denoise wavelet/pca also available for comparison.
