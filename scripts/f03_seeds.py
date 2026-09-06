#!/usr/bin/env python3
"""F03 — Per-seed spread behind every pooled number.
DATA: ALL 3 seeds shown individually (100 eps each). Nothing hidden."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import pv_style as S, pv_data as D
import matplotlib.pyplot as plt, numpy as np
from matplotlib.lines import Line2D
S.apply()
fig, axes = plt.subplots(1, 3, figsize=(S.FULL_W, 3.5), sharey=True)
fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.34, wspace=0.08)
names=[n for _,n in D.VARIANTS]; x=np.arange(len(names)); MK={42:"o",777:"s",123:"^"}
for ax,(b,bn) in zip(axes, D.BENCH):
    for i,(st,nm) in enumerate(D.VARIANTS):
        vals=[]
        for sd in D.SEEDS:
            n,g,c,t=D.outcomes(st,sd,b)
            if n==0: continue
            v=g/n*100; vals.append(v)
            ax.scatter(x[i], v, marker=MK[sd], s=38, facecolor=S.V[nm],
                       edgecolor=S.INK, lw=0.7, zorder=4)
        if vals:
            ax.plot([x[i]]*2,[min(vals),max(vals)], color=S.V[nm], lw=1.8, alpha=0.6, zorder=3)
            ax.scatter(x[i], np.mean(vals), marker="_", s=200, color=S.INK, zorder=5)
    ax.set_title(bn, fontsize=8.0); ax.set_ylim(0,105); ax.grid(axis="y", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=6.4)
axes[0].set_ylabel("Success rate (%)")
S.legend_below(fig, [Line2D([],[],marker=MK[s],color="none",markerfacecolor=S.GREY,
               markeredgecolor=S.INK,markersize=5.2,label=f"Seed {s}") for s in D.SEEDS]+
               [Line2D([],[],marker="_",color=S.INK,markersize=9.0,lw=0,label="Mean")],
               y=0.012, ncol=4)
S.save(fig, "F03_per_seed_spread")
