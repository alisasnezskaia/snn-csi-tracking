"""Grid-cell position classification pooled across multiple capture-rate
sessions (data/raw + mat_loader.py), not just the single-rate data
train_presence_position_grid.py uses (data/raw_captures).

Why: that script's honest (held-out-trial) split still only has ~10 true
independent sessions to draw from -- checking the actual Drive layout (see
conversation) showed every activity's 5 numbered "captures" are 5 repeats
within *one* recording visit, not 5 independent sessions. The different
capture-rate folders (5/10/50/100ms) turned out to be genuinely different
visits (different dates/times, confirmed against Drive), so pooling across
rates gives real additional independent sessions instead of just relabeling
the same ~10 visits. 100ms is excluded (single-antenna raw shape, no phase-
diff features possible); 5ms is excluded (too large to window without real
OOM risk on this machine, ~500MB/trial raw per mat_loader.py's docstring).
10ms is downsampled to an effective 50ms frame spacing before windowing --
its extra temporal resolution isn't the point here, its *different session
dates* are, and downsampling keeps memory/window-count comparable to the
existing 50ms-only experience instead of ~5x larger.

Session = (condition, rate, activity_code) -- the true independent unit
(all numbered captures within one session share the same day/person/
hardware-warmup/furniture state) -- not the individual capture file, unlike
train_presence_position_grid.py's grouping. Only 20 sessions total (2
conditions x 2 rates x 5 activities), still small, but roughly 2x the
independent-session count of the single-rate pipeline.

Positions come from data/processed/trajectories/ (scripts/
build_trajectory_labels.py), in raw uncalibrated pixel space -- unlike
presence_position_dataset's MediaPipe-normalized [0,1] coordinates from a
different video source/extraction method, so this dataset is NOT merged
with data/raw_captures' (the two pixel spaces aren't compatible without a
homography calibration that hasn't been run -- see calibrate_homography.py).
Grid bounds are computed empirically across all included trials, same as
train_grid_motion.py's grid_motion_labels usage.

Reuses: presence_position_dataset.make_perframe_grid_windows (grid-cell
windowing, once positions are normalized to [0,1] via the empirical
bounds), SNNPresencePositionPerFrame (same trunk/heads), preprocessing.
denoise_amplitude + phase_diff_features, dataset.build_datasets_grouped
(session-level split, stratified by activity_code).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from snn_csi_tracking.data import mat_loader
from snn_csi_tracking.data.dataset import build_datasets_grouped
from snn_csi_tracking.data.preprocessing import denoise_amplitude, phase_diff_features
from snn_csi_tracking.data.presence_position_dataset import make_perframe_grid_windows
from snn_csi_tracking.models.presence_position_snn import SNNPresencePositionPerFrame, to_spikes

REPO_ROOT = Path(__file__).parent.parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "raw"
TRAJECTORY_DIR = REPO_ROOT / "data" / "processed" / "trajectories"
CACHE_DIR = REPO_ROOT / "data" / "processed"

CONDITIONS = ("NLoS", "PLoS")
RATES_MS = (10, 50)  # 5ms excluded (OOM risk), 100ms excluded (1 antenna only)
TARGET_RATE_MS = 50  # every rate is downsampled to this effective spacing

GRID_DIM = 3
T_WIN = 64
STRIDE = 32
MAX_NAN_FRAC = 0.3  # skip non-empty trials where pose detection mostly failed

USE_EMPTY_BASELINE = True
DENOISE = None  # None, "wavelet", or "pca"

DELTA_THRESHOLD = 0.3
HIDDEN_1 = 128
HIDDEN_2 = 32
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
NUM_EPOCHS = 60
NUM_WORKERS = 4
SEED = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _downsample_step(rate_ms: int) -> int:
    """How many native frames make up one TARGET_RATE_MS-spaced frame."""
    assert TARGET_RATE_MS % rate_ms == 0, f"{TARGET_RATE_MS}ms not a multiple of {rate_ms}ms"
    return TARGET_RATE_MS // rate_ms


def _discover_sessions() -> list[dict]:
    """One dict per usable (condition, rate, activity_code, capture) trial,
    with its trajectory path and session key attached."""
    trials = []
    n_skipped_bad_gt = 0
    for condition in CONDITIONS:
        for rate_ms in RATES_MS:
            path = DATA_ROOT / condition / f"csi_office_{rate_ms}ms_interframe.mat"
            if not path.exists():
                continue
            for row in mat_loader.load_table_metadata(path):
                code = row["activity_code"]
                traj_path = TRAJECTORY_DIR / f"{condition}_{rate_ms}ms_{code}_{row['filename']}.npy"
                if not traj_path.exists():
                    continue
                if code != "E":
                    positions = np.load(traj_path)
                    if np.isnan(positions[:, 0]).mean() > MAX_NAN_FRAC:
                        n_skipped_bad_gt += 1
                        continue
                trials.append({
                    "condition": condition, "rate_ms": rate_ms, "path": path, "row": row,
                    "traj_path": traj_path, "session_key": f"{condition}_{rate_ms}ms_{code}",
                })
    if n_skipped_bad_gt:
        print(f"skipped {n_skipped_bad_gt} non-empty trials with >{MAX_NAN_FRAC:.0%} missing position detections")
    return trials


def _load_empty_room_baselines(trials: list[dict], denoise: str | None) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    """Per (condition, rate_ms), amplitude mean/std across every empty-room
    ("E") trial -- this data source's counterpart to raw_capture_loader.
    load_empty_room_baseline, since data/raw uses a different loader/schema."""
    baselines = {}
    for condition in CONDITIONS:
        for rate_ms in RATES_MS:
            step = _downsample_step(rate_ms)
            amps = []
            for t in trials:
                if t["condition"] != condition or t["rate_ms"] != rate_ms or t["row"]["activity_code"] != "E":
                    continue
                csi = mat_loader.load_csi(t["path"], t["row"]["csi_key"], t["row"]["row_index"])  # (T, Sub, Ant)
                if csi is None:
                    continue
                csi = csi[::step].transpose(2, 1, 0)  # -> (Ant, Sub, T_downsampled)
                amps.append(denoise_amplitude(np.abs(csi), denoise, freq_axis=1, time_axis=2))
            if amps:
                all_amp = np.concatenate(amps, axis=2).astype(np.float32)
                baselines[(condition, rate_ms)] = (
                    all_amp.mean(axis=2, keepdims=True), all_amp.std(axis=2, keepdims=True)
                )
                print(f"empty-room baseline for {condition}/{rate_ms}ms: {len(amps)} trial(s)")
    return baselines


def compute_features_pooled(
    csi_tsa: np.ndarray, baseline: tuple[np.ndarray, np.ndarray] | None, denoise: str | None
) -> np.ndarray:
    """csi_tsa: (T, Sub, Ant) complex, mat_loader's convention. Returns
    (Chan, Sub, T) float32 -- amplitude (baseline- or self-normalized) +
    cross-antenna phase-difference, same feature composition as
    presence_position_dataset.compute_features's use_phase=True path."""
    csi = csi_tsa.transpose(2, 1, 0)  # (Ant, Sub, T)
    amp = denoise_amplitude(np.abs(csi), denoise, freq_axis=1, time_axis=2)
    if baseline is not None:
        mu, sigma = baseline
    else:
        mu, sigma = amp.mean(axis=2, keepdims=True), amp.std(axis=2, keepdims=True)
    amp_z = ((amp - mu) / (sigma + 1e-8)).astype(np.float32)
    phase = phase_diff_features(csi_tsa).transpose(2, 1, 0).astype(np.float32)  # (Chan, Sub, T)
    return np.concatenate([amp_z, phase], axis=0)


def build_dataset(denoise: str | None, use_empty_baseline: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, cells, present, groups, activity_codes) -- see
    make_perframe_grid_windows for cells/present's meaning. groups is the
    *session* key (condition_rate_activity), not the individual capture --
    see module docstring on why."""
    trials = _discover_sessions()
    print(f"{len(trials)} usable trials across {len({t['session_key'] for t in trials})} sessions")

    baselines = _load_empty_room_baselines(trials, denoise) if use_empty_baseline else {}

    # Bounds pass: trajectories only (cheap), before touching any CSI --
    # grid cells must mean the same physical region for every trial (same
    # convention as train_grid_motion.py's grid_motion_labels usage).
    all_x, all_y = [], []
    for t in trials:
        if t["row"]["activity_code"] == "E":
            continue
        positions = np.load(t["traj_path"])
        valid = ~np.isnan(positions[:, 0])
        if valid.any():
            all_x.append(positions[valid, 0])
            all_y.append(positions[valid, 1])
    x_min, x_max = np.concatenate(all_x).min(), np.concatenate(all_x).max()
    y_min, y_max = np.concatenate(all_y).min(), np.concatenate(all_y).max()
    print(f"floor bounds (pixel space, uncalibrated): x=[{x_min:.0f},{x_max:.0f}] y=[{y_min:.0f},{y_max:.0f}]")

    X_list, cell_list, present_list, groups, activity_codes = [], [], [], [], []
    t_start = time.time()
    for i, t in enumerate(trials, start=1):
        csi = mat_loader.load_csi(t["path"], t["row"]["csi_key"], t["row"]["row_index"])
        if csi is None:
            continue
        step = _downsample_step(t["rate_ms"])
        csi = csi[::step]  # (T_downsampled, Sub, Ant)

        positions = np.load(t["traj_path"])[::step]
        if len(positions) != csi.shape[0]:
            n = min(len(positions), csi.shape[0])
            positions, csi = positions[:n], csi[:n]

        baseline = baselines.get((t["condition"], t["rate_ms"])) if use_empty_baseline else None
        amp_z = compute_features_pooled(csi, baseline, denoise)

        positions_norm = np.stack(
            [(positions[:, 0] - x_min) / (x_max - x_min + 1e-8), (positions[:, 1] - y_min) / (y_max - y_min + 1e-8)],
            axis=1,
        )
        windows = make_perframe_grid_windows(amp_z, positions_norm, GRID_DIM, T_WIN, STRIDE)
        for w, cell_seq, present_seq in windows:
            X_list.append(w)
            cell_list.append(cell_seq)
            present_list.append(present_seq)
            groups.append(t["session_key"])
            activity_codes.append(t["row"]["activity_code"])

        if i % 10 == 0 or i == len(trials):
            print(f"[{i}/{len(trials)}] {t['session_key']}/{t['row']['filename']}: "
                  f"{len(windows)} windows  ({time.time()-t_start:.1f}s elapsed)")

    X = np.stack(X_list, axis=0).astype(np.float32)
    cells = np.stack(cell_list, axis=0)
    present = np.stack(present_list, axis=0)
    groups = np.array(groups)
    activity_codes = np.array(activity_codes)
    print(f"presence rate: {present.mean():.2f}")
    return X, cells, present, groups, activity_codes


def cache_paths(denoise: str | None, use_empty_baseline: bool) -> tuple[Path, Path]:
    tag = f"grid_position_pooled_grid{GRID_DIM}_w{T_WIN}_s{STRIDE}"
    tag += "_ebase" if use_empty_baseline else ""
    tag += f"_{denoise}denoise" if denoise else ""
    return CACHE_DIR / f"{tag}_meta.npz", CACHE_DIR / f"{tag}_X.npy"


def load_or_build_dataset(denoise: str | None, use_empty_baseline: bool):
    meta_path, x_path = cache_paths(denoise, use_empty_baseline)
    if meta_path.exists() and x_path.exists():
        print(f"Loading cached dataset from {x_path}")
        meta = np.load(meta_path, allow_pickle=True)
        X = np.load(x_path, mmap_mode="r")
        return X, meta["cells"], meta["present"], meta["groups"], meta["activity_codes"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    X, cells, present, groups, activity_codes = build_dataset(denoise, use_empty_baseline)
    meta_path, x_path = cache_paths(denoise, use_empty_baseline)
    np.save(x_path, X)
    np.savez(meta_path, cells=cells, present=present, groups=groups, activity_codes=activity_codes)
    print(f"Cached dataset to {x_path}")
    return X, cells, present, groups, activity_codes


def make_dataloaders(X, cells, present, groups, activity_codes, batch_size):
    train_ds, val_ds, test_ds = build_datasets_grouped(
        X, cells, groups, label_dtype=torch.long, extra=present, strata=activity_codes,
    )
    print(f"split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} windows, "
          f"train_sessions={len(set(groups[train_ds.indices]))} "
          f"val_sessions={len(set(groups[val_ds.indices]))} "
          f"test_sessions={len(set(groups[test_ds.indices]))}")
    loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"), persistent_workers=NUM_WORKERS > 0)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        train_ds, test_ds,
    )


def encode(windows: torch.Tensor) -> torch.Tensor:
    return to_spikes(windows, threshold=DELTA_THRESHOLD)


def train_one_epoch(model, loader, optimizer, bce_loss, ce_loss) -> float:
    model.train()
    total_loss, total = 0.0, 0
    for windows, cell_seq, pres_seq in loader:
        windows = windows.to(DEVICE)
        cell_seq = cell_seq.to(DEVICE).permute(1, 0)
        pres_seq = pres_seq.to(DEVICE).permute(1, 0)
        optimizer.zero_grad()
        x_seq = encode(windows)
        pres_pred, cell_logits = model(x_seq)
        loss_presence = bce_loss(pres_pred, pres_seq).mean()
        k = cell_logits.shape[-1]
        ce_per_frame = ce_loss(cell_logits.reshape(-1, k), cell_seq.reshape(-1)).reshape(cell_seq.shape)
        loss_grid = (ce_per_frame * pres_seq).sum() / (pres_seq.sum() + 1e-8)
        loss = loss_presence + loss_grid
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pres_seq.shape[1]
        total += pres_seq.shape[1]
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, bce_loss, ce_loss) -> tuple[float, float, float]:
    model.eval()
    all_pres_pred, all_cell_logits, all_pres, all_cells = [], [], [], []
    for windows, cell_seq, pres_seq in loader:
        windows = windows.to(DEVICE)
        cell_seq = cell_seq.to(DEVICE).permute(1, 0)
        pres_seq = pres_seq.to(DEVICE).permute(1, 0)
        x_seq = encode(windows)
        pres_pred, cell_logits = model(x_seq)
        all_pres_pred.append(pres_pred)
        all_cell_logits.append(cell_logits)
        all_pres.append(pres_seq)
        all_cells.append(cell_seq)
    pres_pred = torch.cat(all_pres_pred, dim=1)
    cell_logits = torch.cat(all_cell_logits, dim=1)
    pres = torch.cat(all_pres, dim=1)
    cells = torch.cat(all_cells, dim=1)

    presence_acc = ((torch.sigmoid(pres_pred) > 0.5).float() == pres).float().mean().item()
    pred_cell = cell_logits.argmax(dim=-1)
    grid_acc = ((pred_cell == cells).float() * pres).sum().item() / (pres.sum().item() + 1e-8)

    k = cell_logits.shape[-1]
    ce_per_frame = ce_loss(cell_logits.reshape(-1, k), cells.reshape(-1)).reshape(cells.shape)
    loss_grid = (ce_per_frame * pres).sum() / (pres.sum() + 1e-8)
    loss = bce_loss(pres_pred, pres).mean().item() + loss_grid.item()
    return loss, presence_acc, grid_acc


def majority_baseline_acc(train_ds, test_ds, k: int) -> tuple[float, int]:
    train_present = train_ds.extra.bool()
    majority_cell = torch.bincount(train_ds.labels[train_present], minlength=k).argmax().item()
    test_present = test_ds.extra.bool()
    test_cells = test_ds.labels[test_present]
    acc = (test_cells == majority_cell).float().mean().item()
    return acc, majority_cell


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--denoise", choices=["none", "wavelet", "pca"], default=(DENOISE or "none"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    denoise = None if args.denoise == "none" else args.denoise

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t_start = time.time()
    k = GRID_DIM * GRID_DIM
    print(f"Pooling rates {RATES_MS} (downsampled to {TARGET_RATE_MS}ms), GRID_DIM={GRID_DIM} ({k} cells), "
          f"USE_EMPTY_BASELINE={USE_EMPTY_BASELINE}, denoise={denoise}, seed={args.seed}...")
    X, cells, present, groups, activity_codes = load_or_build_dataset(denoise, USE_EMPTY_BASELINE)
    print(f"X={X.shape}, cells={cells.shape}, {len(set(groups))} sessions ({time.time()-t_start:.1f}s)")

    train_loader, val_loader, test_loader, train_ds, test_ds = make_dataloaders(
        X, cells, present, groups, activity_codes, BATCH_SIZE
    )

    chance_acc = 1.0 / k
    test_majority_acc, majority_cell = majority_baseline_acc(train_ds, test_ds, k)
    print(f"chance accuracy (1/{k}): {chance_acc:.3f}")
    print(f"majority-cell baseline (always predict train's cell {majority_cell}): test acc={test_majority_acc:.3f}")

    present_mask = train_ds.extra.bool()
    present_cells = train_ds.labels[present_mask]
    class_counts = torch.bincount(present_cells, minlength=k).float()
    print(f"train cell distribution (present frames only): {class_counts.tolist()}")
    class_weights = (class_counts.sum() / (class_counts + 1e-6))
    class_weights = (class_weights / class_weights.mean()).to(DEVICE)

    train_presence_rate = train_ds.extra.float().mean().item()
    pos_weight = torch.tensor([(1 - train_presence_rate) / train_presence_rate]).to(DEVICE)
    print(f"pos_weight (presence) = {pos_weight.item():.3f}")

    in_features = X.shape[1] * X.shape[2]
    model = SNNPresencePositionPerFrame(in_features=in_features, h1=HIDDEN_1, h2=HIDDEN_2, out_dim=k).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    best_val_grid_acc, best_state, best_epoch = -1.0, None, None

    print(f"\nTraining on {DEVICE} for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, bce_loss, ce_loss)
        val_loss, val_presence_acc, val_grid_acc = evaluate(model, val_loader, bce_loss, ce_loss)
        marker = ""
        if val_grid_acc > best_val_grid_acc:
            best_val_grid_acc = val_grid_acc
            best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
            best_epoch = epoch
            marker = "  <- best so far"
        print(
            f"epoch {epoch:2d}/{NUM_EPOCHS}  train_loss={train_loss:.5f}  "
            f"val_presence_acc={val_presence_acc:.3f}  val_grid_acc={val_grid_acc:.3f} (chance={chance_acc:.3f})  "
            f"({time.time()-t0:.1f}s){marker}"
        )

    print(f"\nrestoring best checkpoint from epoch {best_epoch} (val_grid_acc={best_val_grid_acc:.3f})")
    model.load_state_dict(best_state)

    _test_loss, test_presence_acc, test_grid_acc = evaluate(model, test_loader, bce_loss, ce_loss)
    print(f"\nFINAL TEST: presence_acc={test_presence_acc:.3f}, grid_acc={test_grid_acc:.3f}  "
          f"(chance={chance_acc:.3f}, majority-baseline={test_majority_acc:.3f})")
    print(f"total wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
