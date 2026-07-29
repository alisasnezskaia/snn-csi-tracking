"""How much does the CSI signal change from one snapshot to the next, per activity?

A simple bar chart -- one bar per activity, height = typical (median) change
between consecutive frames. Answers "can you tell activities apart just by how
much the signal wiggles?" more honestly than a histogram does (histograms of
this data all look like the same exponential-decay shape at a glance; a bar
chart makes the actual size difference visible).

Run:
    .venv/bin/python scripts/compare_activity_deltas.py [condition] [interframe_ms]
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
rows = mat_loader.load_table_metadata(path)
by_code = {}
for r in rows:
    by_code.setdefault(r["activity_code"], []).append(r)

# usable subcarriers (drop near-zero guard bands, same rule as other scripts)
trials = {}
for code in CODES:
    r = by_code[code][0]
    csi = mat_loader.load_csi(path, r["csi_key"], r["row_index"])
    if csi is not None:
        trials[code] = csi
all_amp = np.concatenate([np.abs(c).mean(axis=2) for c in trials.values()], axis=0)
usable = np.where(all_amp.mean(axis=0) > 0.05 * all_amp.mean(axis=0).max())[0]

# median frame-to-frame |change in amplitude|, per activity
medians = {}
for code, csi in trials.items():
    amp = np.abs(csi).mean(axis=2)[:, usable]  # (T, usable_subcarriers)
    delta = np.abs(np.diff(amp, axis=0))
    medians[code] = np.median(delta)

# sort smallest -> largest so the ranking reads left to right
order = sorted(medians, key=medians.get)
values = [medians[c] for c in order]

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(order, values, color="#4a7fb5", width=0.6)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("typical change between consecutive snapshots\n(median |delta amplitude|)")
ax.set_title(f"{CONDITION} {INTERFRAME_MS}ms -- how much the signal wiggles, by activity")
ax.tick_params(axis="x", labelsize=11)

# legend mapping short codes -> full description, below the chart, so bar
# labels don't collide with each other
caption = "   |   ".join(f"{c} = {mat_loader.ACTIVITY_DESCRIPTIONS[c]}" for c in order)
fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8, wrap=True)
fig.tight_layout()
out_path = OUT_DIR / "05_activity_delta_barchart.png"
fig.savefig(out_path, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_path}")
for c in order:
    print(f"  {c:6s} median={medians[c]:.5f}  ({medians[c]/values[0]:.2f}x the smallest)")
