#!/usr/bin/env python3
"""S02 — System pipeline. Three stages, one idea per layer, minimum width.
NO human-obstacle panel. NO control rate stated (sim 150 ms vs hardware 33.3 ms differ)."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import pv_style as S
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
S.apply()
fig, ax = plt.subplots(figsize=(S.FULL_W, 6.2))
ax.set_xlim(-9, 84); ax.set_ylim(-2, 102); ax.axis("off")
L, R = 2, 82                                   # stage band extent

def stage(y, h, num, label):
    ax.add_patch(Rectangle((L, y), R-L, h, facecolor="#FAFAFA",
                           edgecolor="#D8D8D8", lw=1.0, zorder=0))
    ax.text(-1.5, y+h/2, f"{num}\n{label}", ha="center", va="center", rotation=90,
            fontsize=7.4, fontweight="bold", color="#666666")

# ── 1 SENSING ────────────────────────────────────────────────────────────────
stage(70, 30, "1", "SENSING")
S.block(ax, 10, 90, 64, 7, "360° LiDAR  →  24 sectors of 15°", None, "input")
S.arrow(ax, 42, 90, 42, 86)
S.block(ax,  7, 74, 30, 10, "PA-Obs", "$\\acute{s}_i = s_i^2$  if  $s_i < 0.30$ m", "input")
S.block(ax, 47, 74, 30, 10, "Scan difference", "$\\Delta s_i = s_i(t) - s_i(t\\!-\\!1)$",
        "input", True)
ax.text(62, 71.4, "velocity without ICP or a map", ha="center", fontsize=5.9,
        color="#666666", style="italic")

# ── 2 POLICY ─────────────────────────────────────────────────────────────────
stage(30, 36, "2", "POLICY")
S.arrow(ax, 22, 74, 32, 62); S.arrow(ax, 62, 74, 52, 62)
S.block(ax, 22, 53, 40, 8.5, "PV-STAM attention", "24 sectors  →  48-dim", "novel", True)
S.arrow(ax, 42, 53, 42, 49)
S.block(ax, 22, 41, 40, 7.5, "concatenate $o_{nav}$ (6-dim)", "54-dim state", "process")
S.arrow(ax, 42, 41, 42, 37)
S.block(ax, 22, 32, 40, 8, "SAC actor-critic", "continuous  $(v_{lin},\\, \\omega)$", "policy")
S.block(ax, 66, 40, 13, 12, "HCTM", "delay +\nnoise\nPhase 7", "perturb", fs=8.6, subfs=6.9)
S.arrow(ax, 66, 45, 62, 44.5)

# ── 3 VALIDATION ─────────────────────────────────────────────────────────────
stage(2, 24, "3", "VALIDATION")
S.arrow(ax, 42, 32, 42, 27)
for x0, title, sub, ph in ((8, "Gazebo simulation", "curriculum + benchmarks", "screenshot"),
                           (45, "Physical corridor", "3.3 × 7.6 m  ·  130 trials", "photograph")):
    S.block(ax, x0, 5, 31, 17, title, None, "terminal" if x0 > 40 else "process", fs=9.4)
    ax.text(x0+15.5, 17.0, f"[ {ph} ]", ha="center", fontsize=5.9, color="#AAAAAA", style="italic")
    ax.text(x0+15.5, 8.0, sub, ha="center", fontsize=6.1, color="#555555")

# environment feedback
S.arrow(ax, 22, 36, 5, 36, dashed=True)
S.arrow(ax, 5, 36, 5, 93.5, dashed=True)
S.arrow(ax, 5, 93.5, 10, 93.5, dashed=True)
ax.text(3.4, 66, "next state  ·  reward", rotation=90, ha="center", va="center",
        fontsize=5.9, color="#888888", style="italic")
S.save(fig, "S02_system_pipeline")
