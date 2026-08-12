"""Single-panel validation-RMSE-vs-epoch plot (no presence-accuracy panel),
regenerated from an already-saved rmse_vs_epoch_history_*.npz -- no
retraining, just re-plots the raw per-fold, per-epoch RMSE that
plot_rmse_vs_epoch_3d.py / plot_rmse_vs_epoch.py already saved.

Run:
    .venv/bin/python scripts/plot_rmse_vs_epoch_rmse_only.py
    .venv/bin/python scripts/plot_rmse_vs_epoch_rmse_only.py --history results/rmse_vs_epoch_history_3d.npz --ylabel "Validation RMSE (m, camera frame)" --out rmse_vs_epoch_3d_rmse_only
Output:
    results/figures/<out>.png (and .pdf)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent

MODELS = ["snn", "ann", "lstm"]
LABELS = {"snn": "Conv-SNN (best, cross\\_coherence)", "ann": "Conv-ANN", "lstm": "LSTM"}
COLORS = {"snn": "#2563eb", "ann": "#dc2626", "lstm": "#eab308"}  # same convention as plot_rmse_vs_epoch_3d.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default=str(REPO_ROOT / "results" / "rmse_vs_epoch_history_3dz.npz"),
                         help="which saved per-fold, per-epoch history to plot -- e.g. "
                              "rmse_vs_epoch_history.npz (2D), rmse_vs_epoch_history_3d.npz (pinhole), "
                              "rmse_vs_epoch_history_3dz.npz (no-pinhole, the default here)")
    parser.add_argument("--ylabel", default="Validation RMSE")
    parser.add_argument("--out", default="rmse_vs_epoch_3dz_rmse_only", help="output filename stem")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(args.history)
    epochs = d["epochs"]

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 9,
    })

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for m in MODELS:
        rmse = d[f"{m}_rmse"]  # (n_folds, n_epochs)
        mean, std = rmse.mean(axis=0), rmse.std(axis=0)
        color = COLORS[m]
        ax.plot(epochs, mean, label=LABELS[m], color=color, linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(args.ylabel)
    ax.set_xlim(1, epochs[-1])
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, which="major")
    ax.legend(frameon=False)

    plt.tight_layout()
    out_dir = REPO_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{args.out}.png"
    plt.savefig(png_path, dpi=200)
    plt.savefig(out_dir / f"{args.out}.pdf")
    print(f"saved to {png_path} and .pdf")


if __name__ == "__main__":
    main()
