"""Quick visual check of a saved trajectory array (e.g. /tmp/test_traj.npy
from the extract_pose_trajectory smoke test) -- same 3-panel style as
results/figures/08_video_trajectory_pose.png for easy eyeballing.

Run:
    .venv/bin/python scripts/plot_test_trajectory.py /tmp/test_traj.npy [fps]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

traj_path = Path(sys.argv[1])
fps = float(sys.argv[2]) if len(sys.argv) > 2 else 28.25

positions = np.load(traj_path)
x, y = positions[:, 0], positions[:, 1]
t = np.arange(len(positions)) / fps
valid = ~np.isnan(x)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(x, y, "-", alpha=0.4, linewidth=1)
axes[0].scatter(x[valid], y[valid], c=t[valid], cmap="viridis", s=4)
axes[0].invert_yaxis()
axes[0].set_xlabel("x (pixels)")
axes[0].set_ylabel("y (pixels)")
axes[0].set_title("path (color = time)")

axes[1].plot(t, x, label="x", linewidth=1)
axes[1].plot(t, y, label="y", linewidth=1)
axes[1].set_xlabel("time (s)")
axes[1].set_ylabel("position (pixels)")
axes[1].set_title("position vs time")
axes[1].legend()

fig.suptitle(f"Robustified pose trajectory -- {traj_path.name}")
fig.tight_layout()
out_path = Path(__file__).parent.parent / "results" / "figures" / "09_video_trajectory_robust.png"
fig.savefig(out_path, dpi=130)
print(f"wrote {out_path}")
