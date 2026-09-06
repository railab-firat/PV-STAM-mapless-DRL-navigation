#!/usr/bin/env python3
"""F00 — palette reference sheet. Not a paper figure; a check that every colour
used anywhere comes from pv_style.py."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import pv_style as S
import matplotlib.pyplot as plt
S.apply()
groups=[("Variant colours", [(n, S.V[n]) for n in S.ORDER]),
        ("Outcome colours", [("Goal reached", S.OUT["goal"]),
                             ("Timeout", S.OUT["timeout"]),
                             ("Collision", S.OUT["collision"])]),
        ("Role colours", [("Ablated condition (No-ω)", S.ORANGE),
                          ("Single-series bars", S.SLATE),
                          ("Receding / static", S.BLUE),
                          ("Rotation-artefact edge", S.RED_DK)])]
rows=sum(len(g[1]) for g in groups)+len(groups)
fig,ax=plt.subplots(figsize=(7.4, 0.42*rows+0.8)); ax.axis("off")
y=rows
for title,items in groups:
    ax.text(0, y, title, fontsize=8.4, fontweight="bold", va="center"); y-=1
    for name,col in items:
        ax.add_patch(plt.Rectangle((0.30, y-0.34), 0.55, 0.68, facecolor=col,
                                   edgecolor=S.INK, lw=0.8))
        ax.text(0.95, y, name, fontsize=7.4, va="center")
        ax.text(4.55, y, col, fontsize=6.9, va="center", family="monospace", color="#666666")
        y-=1
ax.set_xlim(0,5.6); ax.set_ylim(-0.4, rows+0.6)
S.save(fig, "F00_palette_reference")
