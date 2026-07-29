"""Compare the 4 interframe rates (5/10/50/100ms) side by side, same activity.

Plain matplotlib. Shows exactly what differs about the 100ms files: fewer
antennas, fewer subcarriers, and a different amplitude-vs-subcarrier shape.

Run:
    .venv/bin/python scripts/compare_interframe_rates.py [condition] [activity_code]

Defaults to PLoS, "E" (empty room -- cleanest signal, no movement to confound
the comparison).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from snn_csi_tracking.data import mat_loader

CONDITION = sys.argv[1] if len(sys.argv) > 1 else "PLoS"
ACTIVITY_CODE = sys.argv[2] if len(sys.argv) > 2 else "E"
RATES = [5, 10, 50, 100]

OUT_DIR = Path(__file__).parent.parent / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- load one trial of the same activity at each rate ---
trials = {}  # rate -> (T, NumSubcarriers, NumAntennas) complex
for rate in RATES:
    path = Path(__file__).parent.parent / "data" / "raw" / CONDITION / f"csi_office_{rate}ms_interframe.mat"
    rows = mat_loader.load_table_metadata(path)
    row = next(r for r in rows if r["activity_code"] == ACTIVITY_CODE)
    csi = mat_loader.load_csi(path, row["csi_key"], row["row_index"])
    trials[rate] = csi
    print(f"{rate:>4}ms: raw csi shape (T, NumSubcarriers, NumAntennas) = {csi.shape}")

# --- row 1: amplitude vs subcarrier index (own x-axis per rate -- counts differ) ---
# --- row 2: amplitude heatmap, subcarrier x time (own axes per rate) ---
fig, axes = plt.subplots(2, len(RATES), figsize=(16, 6))

for col, rate in enumerate(RATES):
    csi = trials[rate]
    amp = np.abs(csi).mean(axis=2)  # (T, NumSubcarriers), antenna-averaged

    ax = axes[0, col]
    ax.plot(amp.mean(axis=0))
    ax.set_title(f"{rate}ms\nNumSubcarriers={csi.shape[1]}, NumAntennas={csi.shape[2]}", fontsize=9)
    ax.set_xlabel("subcarrier index")
    if col == 0:
        ax.set_ylabel("mean |CSI| amplitude\n(avg over time)")

    ax = axes[1, col]
    im = ax.imshow(amp[:300].T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("frame")
    if col == 0:
        ax.set_ylabel("subcarrier index\n(first 300 frames)")

fig.suptitle(f"{CONDITION}, activity={ACTIVITY_CODE} -- same activity, 4 interframe rates")
fig.tight_layout()
out_path = OUT_DIR / "04_interframe_rate_comparison.png"
fig.savefig(out_path, dpi=130)
plt.close(fig)
print(f"\nwrote {out_path}")
