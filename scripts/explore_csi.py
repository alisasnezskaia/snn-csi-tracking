"""Plot amplitude/phase diagnostics for one trial per activity.

Plain matplotlib, no theming, no HTML. Run it, look at the PNGs in results/figures/:

    .venv/bin/python scripts/explore_csi.py [condition] [interframe_ms]

Defaults to PLoS, 50ms.

For "how much does the signal change between activities" specifically, use
scripts/compare_activity_deltas.py instead -- a per-activity histogram (which
this script used to produce) looks the same shape for every activity at a
glance and hides the actual size difference; a bar chart of the typical
change shows it directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from snn_csi_tracking.data import mat_loader

CONDITION = sys.argv[1] if len(sys.argv) > 1 else "PLoS"
INTERFRAME_MS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
CODES = list(mat_loader.ACTIVITY_CODES)

OUT_DIR = Path(__file__).parent.parent / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

path = Path(__file__).parent.parent / "data" / "raw" / CONDITION / f"csi_office_{INTERFRAME_MS}ms_interframe.mat"

# --- load one trial per activity ---
rows = mat_loader.load_table_metadata(path)
by_code = {}
for r in rows:
    by_code.setdefault(r["activity_code"], []).append(r)

trials = {}  # code -> (T, NumSubcarriers, NumAntennas) complex
for code in CODES:
    r = by_code[code][0]
    csi = mat_loader.load_csi(path, r["csi_key"], r["row_index"])
    if csi is not None:
        trials[code] = csi

# usable subcarriers: drop near-zero guard bands (empirical threshold, see mat_loader docstring)
all_amp = np.concatenate([np.abs(c).mean(axis=2) for c in trials.values()], axis=0)
sc_mean = all_amp.mean(axis=0)
usable = np.where(sc_mean > 0.05 * sc_mean.max())[0]
print(f"usable subcarriers: {usable.min()}-{usable.max()} (n={len(usable)} of {len(sc_mean)})")

# --- Figure 1: amplitude vs subcarrier index, one panel per activity ---
fig, axes = plt.subplots(1, len(trials), figsize=(15, 3), sharey=True)
for ax, code in zip(axes, trials):
    amp = np.abs(trials[code]).mean(axis=2)  # antenna-averaged
    snaps = np.linspace(0, len(amp) - 1, 15).astype(int)
    for t in snaps:
        ax.plot(usable, amp[t, usable], alpha=0.3, linewidth=0.8)
    ax.plot(usable, amp[:, usable].mean(axis=0), color="black", linewidth=1.5)
    ax.set_title(f"{code}: {mat_loader.ACTIVITY_DESCRIPTIONS[code]}", fontsize=8)
    ax.set_xlabel("subcarrier index")
axes[0].set_ylabel("|CSI| amplitude")
fig.suptitle(f"{CONDITION} {INTERFRAME_MS}ms — amplitude vs subcarrier (15 snapshots + mean)")
fig.tight_layout()
fig.savefig(OUT_DIR / "01_amplitude_vs_subcarrier.png", dpi=130)
plt.close(fig)

# --- Figure 2: amplitude heatmap, subcarrier x time ---
fig, axes = plt.subplots(1, len(trials), figsize=(15, 3.5))
vmax = np.percentile(all_amp[:, usable], 99)
for ax, code in zip(axes, trials):
    amp = np.abs(trials[code]).mean(axis=2)[:600, usable]
    im = ax.imshow(amp.T, aspect="auto", origin="lower", vmin=0, vmax=vmax, cmap="viridis")
    ax.set_title(code, fontsize=9)
    ax.set_xlabel("frame")
axes[0].set_ylabel("usable subcarrier")
fig.colorbar(im, ax=axes, shrink=0.8)
fig.suptitle(f"{CONDITION} {INTERFRAME_MS}ms — amplitude heatmap (first 600 frames)")
fig.savefig(OUT_DIR / "02_amplitude_heatmap.png", dpi=130)
plt.close(fig)

# --- Figure 3: phase stability check (adjacent-subcarrier and frame-to-frame) ---
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
code0 = CODES[0]
phase = np.angle(trials[code0][:, :, 0])  # (T, NumSubcarriers), antenna 0
t0 = phase[0, usable]
axes[0].plot(usable, np.unwrap(t0))
axes[0].set_title("phase vs subcarrier (t=0, unwrapped)", fontsize=9)
axes[0].set_xlabel("subcarrier index")

fixed_sc = phase[:300, usable[len(usable) // 2]]
axes[1].plot(np.unwrap(fixed_sc))
axes[1].set_title("phase vs frame (fixed subcarrier, unwrapped)", fontsize=9)
axes[1].set_xlabel("frame")
fig.suptitle(f"{CONDITION} {INTERFRAME_MS}ms activity={code0} — raw phase sanity check")
fig.tight_layout()
fig.savefig(OUT_DIR / "03_phase_check.png", dpi=130)
plt.close(fig)

print(f"wrote 3 PNGs to {OUT_DIR}")
for code in trials:
    amp = np.abs(trials[code]).mean(axis=2)[:, usable]
    print(f"{code:6s} amp mean={amp.mean():.4f} std={amp.std():.4f}")
