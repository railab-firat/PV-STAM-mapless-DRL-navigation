#!/usr/bin/env python3
"""S04 — HCTM pipeline. Four layers, left to right, one perturbation each.
NOTE: the 65 ms and sigma = 0.02 values have NO traceable calibration in the
repository; measured commanded-to-observed lag is ~152 ms. Shown as published."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import pv_style as S
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
S.apply()
fig, ax = plt.subplots(figsize=(S.FULL_W, 1.9))
ax.set_xlim(0,110); ax.set_ylim(0,30); ax.axis("off")
ax.add_patch(Rectangle((17,4), 76, 20, facecolor="#FAFAFA", edgecolor="#D8D8D8", lw=1.0, zorder=0))
ax.text(55, 26.6, "Hardware-Calibrated Training Mode  —  Phase 7 only",
        ha="center", fontsize=8.0, fontweight="bold")
S.block(ax,  1, 11, 15, 8, "DRL agent", "SAC policy", "policy")
S.arrow(ax, 16, 15, 21, 15, label="$a(t)$")
S.block(ax, 21, 11, 20, 8, "Latency buffer", "65 ms FIFO", "perturb", True)
S.arrow(ax, 41, 15, 46, 15, label="$a(t-\\tau)$")
S.block(ax, 46, 11, 20, 8, "Gazebo", "ideal plant", "process")
S.arrow(ax, 66, 15, 71, 15, label="$s^*(t)$")
S.block(ax, 71, 11, 21, 8, "Velocity noise", "Gaussian, $\\sigma$ = 0.02 m/s", "perturb", True)
S.arrow(ax, 92, 15, 100, 15)
ax.text(101, 15, "$s(t)$", fontsize=7.5, va="center")
S.arrow(ax, 81, 11, 8.5, 11, dashed=True, curve=0.16)
ax.text(46, 5.4, "noisy state fed back to the agent", ha="center", fontsize=6.2,
        color="#888888", style="italic")
S.save(fig, "S04_hctm_pipeline")
