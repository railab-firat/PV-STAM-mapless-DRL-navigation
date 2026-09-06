#!/usr/bin/env python3
"""S01 — PV-STAM module. LAYERED, minimal, contribution-first.
Verified line-by-line against sac_stam.py:274-308. Shows the two things that are
new (dual-channel input, sector attention) and the one step the published figure
got wrong (flatten+compress, NOT mean pool). Everything else is one line."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import pv_style as S
import matplotlib.pyplot as plt
S.apply()
fig, ax = plt.subplots(figsize=(S.COL_W, 6.0))
ax.set_xlim(0, 58); ax.set_ylim(-3, 96); ax.axis("off")
X, W = 8, 40; CX = X + W/2
def L(y, h, t, sub, role, hi=False): S.block(ax, X, y, W, h, t, sub, role, hi)

L(76, 11, "24 LiDAR sectors × 2 channels", "distance $\\acute{s}_i$   ·   change $\\Delta s_i$",
  "input", True)
S.note(ax, X+W+1.5, 81.5, "position AND motion,\nper sector")
S.arrow(ax, CX, 76, CX, 72)
L(63, 9, "$W_{in}$ — shared projection", "2 → 16 per sector", "process")
S.arrow(ax, CX, 63, CX, 57)
L(48, 9, "+ learned positional encoding", "sector identity preserved", "process")
S.note(ax, X+W+1.5, 52.5, "384 learned\nparameters")
S.arrow(ax, CX, 48, CX, 42)
L(30, 12, "Multi-head sector attention", "2 heads × 8 dims\nsoftmax($QK^{\\top}/\\sqrt{8}$)·V  →  24 × 16",
  "novel", True)
S.note(ax, X+W+1.5, 36, "each sector weighs\nevery other sector")
S.arrow(ax, CX, 30, CX, 24)
L(12, 12, "Flatten → compress", "24 × 16 = 384  →  48\n18,480 par. — 92% of the module",
  "novel", True)
S.note(ax, X+W+1.5, 18, "order-preserving:\nkeeps WHICH sector\nwas attended to")
S.arrow(ax, CX, 12, CX, 7)
L(0, 7, "48-dim embedding  +  $o_{nav}$ (6)", "54-dim state  ->  SAC actor-critic", "policy")
ax.text(CX, 93.5, "PV-STAM  ·  19,968 parameters", ha="center", fontsize=9.2, fontweight="bold")
ax.text(CX, 90.4, "(19,952 for the 2-channel recurrent variant)", ha="center", fontsize=6.4, color="#666666")
S.save(fig, "S01_pvstam_module")
