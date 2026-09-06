#!/usr/bin/env python3
"""S03 — Seven-phase curriculum. Clean spacing without text overlap.
Matches published palette and layout."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pv_style as S
import matplotlib.pyplot as plt

S.apply()

PH = [
    ("1", "0 obstacles", "SR \u2265 70%", "input"),
    ("2", "0 obstacles", "SR \u2265 70%", "input"),
    ("3", "0 obstacles", "SR \u2265 65%", "input"),
    ("4", "0 obstacles", "SR \u2265 60%", "input"),
    ("5", "6 obstacles\n0.08 m/s", "SR \u2265 60%", "novel"),
    ("6", "9 obstacles\n0.12 m/s", "SR \u2265 65%", "novel"),
    ("7", "15 obstacles\n0.18 m/s", "terminal", "terminal")
]

fig, ax = plt.subplots(figsize=(S.FULL_W, 1.8))
ax.set_xlim(0, 126)
ax.set_ylim(0, 28)
ax.axis("off")

w, gap, y, h = 15.4, 2.4, 5.5, 15.0

for i, (n, body, thr, role) in enumerate(PH):
    x = 1 + i * (w + gap)
    # Draw background box without title text inside block()
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.30,rounding_size=0.9",
        facecolor=S.FILL[role], edgecolor=S.EDGE[role],
        linewidth=2.2 if n == "7" else 1.1, zorder=3
    ))
    
    # 1. Title at top of box
    ax.text(x + w / 2, y + h * 0.76, f"Phase {n}", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#1A1A1A", zorder=4)
    
    # 2. Obstacle body in center of box
    ax.text(x + w / 2, y + h * 0.45, body, ha="center", va="center",
            fontsize=6.8, color="#333333", zorder=4)
    
    # 3. Threshold at bottom of box
    ax.text(x + w / 2, y + h * 0.16, thr, ha="center", va="center",
            fontsize=6.2, color="#555555", style="italic", zorder=4)
    
    if i < 6:
        S.arrow(ax, x + w, y + h * 0.62, x + w + gap, y + h * 0.62)
        S.arrow(ax, x + w + gap, y + h * 0.26, x + w, y + h * 0.26, dashed=True)

ax.text(35.5, 23.8, "static navigation", ha="center", fontsize=8.0, color="#4C72B0", fontweight="bold")
ax.text(90.0, 23.8, "dynamic obstacles", ha="center", fontsize=8.0, color="#A8822F", fontweight="bold")
ax.text(63, 1.8, "solid \u2192 promotion when the ESR threshold is met      "
        "dashed \u2192 demotion when rolling ESR falls 10 pp below it",
        ha="center", fontsize=6.5, color="#555555")

S.save(fig, "S03_curriculum")
print("Successfully generated clean S03_curriculum flowchart with zero text overlap.")
