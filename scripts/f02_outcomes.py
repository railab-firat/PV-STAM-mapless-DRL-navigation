#!/usr/bin/env python3
"""F02 — Episode outcome composition per benchmark.
DATA: ALL 3 seeds pooled, n 300 per cell.
LABELS: EVERY segment labelled; segments < 9% placed outside the bar."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import pv_style as S, pv_data as D
import matplotlib.pyplot as plt, numpy as np
S.apply()
fig, axes = plt.subplots(1, 3, figsize=(S.FULL_W, 3.3), sharey=True)
fig.subplots_adjust(left=0.135, right=0.985, top=0.90, bottom=0.24, wspace=0.08)
names=[n for _,n in D.VARIANTS]; y=np.arange(len(names))[::-1]
for ax,(b,bn) in zip(axes, D.BENCH):
    for i,(st,nm) in enumerate(D.VARIANTS):
        n,g,c,t = D.pooled(st,b)
        if n==0: continue
        gp,tp,cp = g/n*100, t/n*100, c/n*100
        left=0.0
        for key,val in (("goal",gp),("timeout",tp),("collision",cp)):
            if val<=0: continue
            ax.barh(y[i], val, left=left, color=S.OUT[key], edgecolor="white", lw=0.8, zorder=3)
            S.stacked_label(ax, val, left+val/2, y[i], True, S.OUT[key])
            left+=val
    ax.set_xlim(0,100); ax.set_xticks([0,20,40,60,80,100])
    ax.set_title(bn, fontsize=8.0); ax.set_xlabel("Episodes (%)")
    ax.set_yticks(y); ax.grid(axis="x", zorder=0)
axes[0].set_yticklabels(names, fontsize=7.2)
S.legend_below(fig, S.outcome_handles(), y=0.012)
S.save(fig, "F02_outcome_composition")
