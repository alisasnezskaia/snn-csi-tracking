"""Publication-ready SNN vs ANN vs LSTM total energy-per-window comparison
-- single clean panel (the per-layer breakdown was dropped as redundant
with the results table), using the exact numbers from
estimate_energy_presence_position.py and estimate_energy_lstm.py's latest
runs (measured from real spike rates on real held-out CSI test data for
the SNN/ANN; closed-form MAC count for the LSTM, which has no sparsity to
measure -- see that script's docstring). Real LaTeX text rendering to
match the paper's fonts exactly.

Run:
    .venv/bin/python scripts/plot_energy_comparison.py
Output: results/figures/energy_comparison.png (and .pdf)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
})

# From estimate_energy_presence_position.py's latest run (T=64 timesteps, per example):
TOTAL_SNN = 122.22 + 5.99 + 0.08 + 1.22 + 1.84  # 131.35 nJ
TOTAL_ANN = 308700.77 + 1205.86 + 9.42 + 150.73 + 9.42  # 310076.21 nJ
# From estimate_energy_lstm.py's latest run:
TOTAL_LSTM = 1260295.78


def main():
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    labels = ["SNN", "ANN", "LSTM"]
    totals = [TOTAL_SNN, TOTAL_ANN, TOTAL_LSTM]
    colors = ["#2563eb", "#dc2626", "#7c3aed"]

    ax.bar(labels, totals, color=colors, width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Energy per window (nJ, log scale)")

    ax.set_ylim(top=TOTAL_LSTM * 10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, which="major")

    plt.tight_layout()
    out_dir = REPO_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "energy_comparison.png", dpi=200)
    plt.savefig(out_dir / "energy_comparison.pdf")
    print(f"saved to {out_dir / 'energy_comparison.png'} and .pdf")


if __name__ == "__main__":
    main()
