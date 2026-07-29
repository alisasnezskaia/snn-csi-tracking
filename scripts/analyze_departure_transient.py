"""What does the CSI signal actually do around the moment someone leaves the
room? Diagnostic for the "presence prediction lags/bounces after departure"
behavior noticed in the per-frame model: is the raw CSI itself still
"active" for a while after departure (a real multipath/reverberation
effect), or does it settle down immediately and the lag is purely coming
from the model (LIF membrane persistence, window smearing)?

Plots, for one trial (default: PLoS_L_capture5 -- the trial already used
for visualization elsewhere, an "enters, sits, stands, leaves" activity so
it has a real single departure point near the end):
  1. z-scored CSI amplitude (subcarrier x time heatmap, antenna 0)
  2. spike rate over time -- the literal to_spikes() signal the model sees
  3. ground-truth presence (from the cached trajectory)
with a vertical line at the last present->absent transition.

Run:
    .venv/bin/python scripts/analyze_departure_transient.py
    .venv/bin/python scripts/analyze_departure_transient.py --trial NLoS_L_capture3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from snn_csi_tracking.data.presence_position_dataset import resample_to_n
from snn_csi_tracking.data.raw_capture_loader import parse_csi_file
from snn_csi_tracking.models.presence_position_snn import to_spikes

REPO_ROOT = Path(__file__).parent.parent
RAW_ROOT = REPO_ROOT / "data" / "raw_captures"
TRAJECTORY_CACHE_DIR = REPO_ROOT / "data" / "processed" / "presence_position_trajectories"
OUT_PATH = REPO_ROOT / "results" / "figures" / "13_departure_transient.png"

RATE_MS = 50
DELTA_THRESHOLD = 0.3
CONTEXT_FRAMES = 100  # +-5s at 50ms, for the zoomed-in panel


def last_departure_index(present: np.ndarray) -> int | None:
    """Index of the last present(1)->absent(0) transition that then stays
    absent through the end of the trial. None if the trial never has one
    (e.g. an empty-room trial, or the person is present at the very end)."""
    if present[-1]:
        return None
    for i in range(len(present) - 1, 0, -1):
        if present[i - 1] and not present[i]:
            return i
    return None


def empty_room_baseline(condition: str) -> np.ndarray | None:
    """Spike rate across every frame of every genuine empty-room ("E") trial
    for this condition -- the actual ground truth for "nobody's there",
    independent of any departure-transition guess."""
    rates = []
    group_dir = RAW_ROOT / f"{condition}_E"
    if not group_dir.is_dir():
        return None
    for csi_path in sorted(group_dir.glob("capture*.mat")):
        csi, _ = parse_csi_file(csi_path)
        if csi.shape[2] == 0:
            continue
        amp = np.abs(csi)
        mu, sigma = amp.mean(axis=2, keepdims=True), amp.std(axis=2, keepdims=True)
        amp_z = (amp - mu) / (sigma + 1e-8)
        spikes = to_spikes(torch.tensor(amp_z[None]), threshold=DELTA_THRESHOLD)
        rates.append(spikes[:, 0, :].mean(dim=1).numpy())
    if not rates:
        return None
    return np.concatenate(rates)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial", default="PLoS_L_capture5")
    args = parser.parse_args()

    condition, activity, capture_name = args.trial.split("_", 2)
    csi_path = RAW_ROOT / f"{condition}_{activity}" / f"{capture_name}.mat"
    traj_path = TRAJECTORY_CACHE_DIR / f"{args.trial}.npy"

    print(f"Loading {csi_path}...")
    csi, _timestamps = parse_csi_file(csi_path)
    amp = np.abs(csi)
    mu, sigma = amp.mean(axis=2, keepdims=True), amp.std(axis=2, keepdims=True)
    amp_z = (amp - mu) / (sigma + 1e-8)
    n_frames = amp_z.shape[2]

    trajectory = np.load(traj_path)
    positions = resample_to_n(trajectory, n_frames)
    present = ~np.isnan(positions[:, 0])

    departure = last_departure_index(present)
    if departure is None:
        print(f"No clean single departure found in {args.trial} (present[-1]={present[-1]}) -- "
              "pick a different --trial (e.g. an 'L' activity trial).")
        return
    print(f"Departure at frame {departure}/{n_frames} ({departure * RATE_MS / 1000:.1f}s into the trial)")

    spikes = to_spikes(torch.tensor(amp_z[None]), threshold=DELTA_THRESHOLD)  # (T, 1, 3072)
    spike_rate = spikes[:, 0, :].mean(dim=1).numpy()  # fraction of channels firing, per frame

    before = spike_rate[max(0, departure - CONTEXT_FRAMES):departure]
    after_near = spike_rate[departure:departure + CONTEXT_FRAMES]
    after_far = spike_rate[departure + CONTEXT_FRAMES:departure + 2 * CONTEXT_FRAMES]
    tail = spike_rate[-CONTEXT_FRAMES:]  # last ~5s of the trial -- as "settled" as this trial gets
    print(f"spike rate -- {CONTEXT_FRAMES * RATE_MS/1000:.0f}s before departure: {before.mean():.4f}")
    print(f"spike rate -- 0-{CONTEXT_FRAMES * RATE_MS/1000:.0f}s after departure:  {after_near.mean():.4f}")
    if len(after_far):
        print(f"spike rate -- {CONTEXT_FRAMES*RATE_MS/1000:.0f}-{2*CONTEXT_FRAMES*RATE_MS/1000:.0f}s after departure: {after_far.mean():.4f}")
    print(f"spike rate -- last {CONTEXT_FRAMES*RATE_MS/1000:.0f}s of the trial (tail): {tail.mean():.4f}")

    baseline_rate = empty_room_baseline(condition)
    if baseline_rate is not None:
        print(f"spike rate -- genuine empty-room baseline ({condition}_E, all captures): "
              f"mean={baseline_rate.mean():.4f}  std={baseline_rate.std():.4f}")
        print(f"tail vs baseline: tail is {(tail.mean() - baseline_rate.mean()) / baseline_rate.std():+.2f} "
              "std away from the empty-room mean")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, height_ratios=[2, 1, 0.5])
    time_s = np.arange(n_frames) * RATE_MS / 1000

    axes[0].imshow(amp_z[0], aspect="auto", origin="lower", extent=[time_s[0], time_s[-1], 0, amp_z.shape[1]], cmap="viridis")
    axes[0].set_ylabel("subcarrier")
    axes[0].set_title(f"{args.trial}: z-scored amplitude (antenna 0)")

    axes[1].plot(time_s, spike_rate, linewidth=0.8)
    axes[1].set_ylabel("spike rate")
    axes[1].set_title(f"fraction of channels spiking (threshold={DELTA_THRESHOLD}) -- what the model actually sees")
    if baseline_rate is not None:
        b_mean, b_std = baseline_rate.mean(), baseline_rate.std()
        axes[1].axhspan(b_mean - b_std, b_mean + b_std, color="green", alpha=0.15, label="empty-room baseline (+-1 std)")
        axes[1].axhline(b_mean, color="green", linewidth=1, linestyle=":")
        axes[1].legend(loc="upper right")

    axes[2].fill_between(time_s, present.astype(float), step="mid", alpha=0.6)
    axes[2].set_ylabel("present")
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_xlabel("time (s)")

    for ax in axes:
        ax.axvline(departure * RATE_MS / 1000, color="red", linestyle="--", linewidth=1, label="departure")
    axes[0].legend(loc="upper right")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
