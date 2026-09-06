#!/usr/bin/env python3
"""Fig 14 — the moving obstacle removes the stall. Real /cmd_vel from trials."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pv_style as S
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
S.apply()

d = np.load(os.path.expanduser("~/Desktop/PVSTAM_FIGURES_FINAL/data/bags.npz"))
STATIC = "v8_s777_scen2_rep5_20260818_010532"      # representative stall trial
MOVING = "v8_s777_scen3_rep6_20260820_010646"      # representative moving trial

fig = plt.figure(figsize=(S.FULL_W, 2.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.85], wspace=0.30)
ax0, ax1, ax2 = (fig.add_subplot(gs[i]) for i in range(3))
G = S.GREEN

# ---- (a) static: the stall ----
ct, cl = d[f"{STATIC}|ct"], d[f"{STATIC}|cl"]
ot, ox, oy = d[f"{STATIC}|ot"], d[f"{STATIC}|ox"], d[f"{STATIC}|oy"]
dist = np.hypot(ox - ox[0], oy - oy[0]); mx = np.maximum.accumulate(dist)
tstall = ot[np.where(np.diff(mx) > 1e-3)[0][-1] + 1]
ax0.axvspan(ct[-1] - 91.3, ct[-1], color=S.RED, alpha=0.10, zorder=1)
ax0.plot(ct, cl, color=G, lw=0.9, zorder=3)
ax0.axvline(tstall, color="#333333", ls="--", lw=1.0, zorder=4)
ax0.text(tstall + 3, 0.235, f"last forward progress\nt = {tstall:.1f} s, x = 2.78 m",
         fontsize=5.6, color="#333333", va="top")
ax0.annotate("zero forward command\nin all 2,746 samples\nof the final 91.3 s",
             xy=(ct[-1] - 45, 0.004), xytext=(ct[-1] - 92, 0.105),
             fontsize=6.0, color=S.RED, ha="left",
             arrowprops=dict(arrowstyle="->", color=S.RED, lw=1.0))
ax0.set_title("(a) Static scenario — stall", fontsize=7.6)
ax0.set_xlabel("Time (s)"); ax0.set_ylabel("Commanded $v_{lin}$ (m s$^{-1}$)")
ax0.set_ylim(-0.01, 0.28); ax0.set_xlim(0, ct[-1]); ax0.grid(zorder=0)

# ---- (b) moving: stall abolished ----
ct2, cl2 = d[f"{MOVING}|ct"], d[f"{MOVING}|cl"]
ax1.plot(ct2, cl2, color=G, lw=0.9, zorder=3)
ax1.set_title("(b) Moving-obstacle scenario — no stall", fontsize=7.6)
ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Commanded $v_{lin}$ (m s$^{-1}$)")
ax1.set_ylim(-0.01, 0.28); ax1.set_xlim(0, ct2[-1]); ax1.grid(zorder=0)
ax1.text(0.97, 0.94, "forward command active 79.6%\nmean 0.129 m s$^{-1}$",
         transform=ax1.transAxes, ha="right", va="top", fontsize=6.0, color="#333333")
ax0.text(0.97, 0.94, "forward command active 16.3%\nmean 0.023 m s$^{-1}$",
         transform=ax0.transAxes, ha="right", va="top", fontsize=6.0, color="#333333")

# ---- (c) outcomes ----
cats = ["Success", "Timeout", "Threshold\nviolation"]
static_v = [10.0, 90.0, 0.0]; moving_v = [55.0, 0.0, 45.0]
x = np.arange(3); w = 0.36
ax2.bar(x - w/2, static_v, w, color=G, edgecolor="#333333", lw=1.0, zorder=3, label="Static (n=10)")
ax2.bar(x + w/2, moving_v, w, color=G, alpha=0.75, hatch="///", edgecolor="#333333",
        lw=1.0, zorder=3, label="Moving (n=20)")
for xi, (a, b) in enumerate(zip(static_v, moving_v)):
    ax2.text(xi - w/2, a + 2, f"{a:.0f}%", ha="center", fontsize=5.6)
    ax2.text(xi + w/2, b + 2, f"{b:.0f}%", ha="center", fontsize=5.6)
ax2.set_xticks(x); ax2.set_xticklabels(cats, fontsize=6.4)
ax2.set_ylim(0, 122); ax2.set_ylabel("Proportion of trials (%)")
ax2.set_title("(c) Outcome distribution", fontsize=7.6)
ax2.grid(axis="y", zorder=0)
ax2.legend(frameon=False, fontsize=7.0, loc="upper center",
           ncol=2, columnspacing=1.0, handlelength=1.4, bbox_to_anchor=(0.5, 1.02))

fig.subplots_adjust(bottom=0.24)
S.save(fig, "F12_stall_removal")
