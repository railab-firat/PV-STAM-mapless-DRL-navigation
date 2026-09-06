#!/usr/bin/env python3
"""F01 — Zero-shot benchmark success rate, 7 variants x 3 benchmarks.
DATA: ALL 3 seeds x 100 episodes pooled = n 300 per cell. No seed omitted.
LABELS: every bar labelled (policy: all or none)."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import pv_style as S, pv_data as D
import matplotlib.pyplot as plt, numpy as np
S.apply()
fig, axes = plt.subplots(1, 3, figsize=(S.FULL_W, 3.3), sharey=True)
fig.subplots_adjust(left=0.23, right=0.97, top=0.88, bottom=0.18, wspace=0.12)
names=[n for _,n in D.VARIANTS]; y=np.arange(len(names))[::-1]
for ax,(b,bn) in zip(axes, D.BENCH):
    for i,(st,nm) in enumerate(D.VARIANTS):
        n,g,c,t = D.pooled(st,b)
        if n==0: continue
        val,lo,hi = D.wilson(g,n)
        ax.barh(y[i], val, height=0.66, color=S.V[nm], edgecolor=S.INK, lw=0.7,
                zorder=3, xerr=[[val-lo],[hi-val]], error_kw=dict(ecolor="#1A1A1A", elinewidth=1.0, capsize=2.4, capthick=1.0, zorder=5))
        ax.text(hi+2.0, y[i], f"{val:.1f}", va="center", ha="left", fontsize=6.6, color="#333333")
    ax.set_xlim(0,112); ax.set_xticks([0,20,40,60,80,100])
    ax.set_title(bn, fontsize=8.2, pad=6); ax.set_xlabel("Success rate (%)", fontsize=7.8)
    ax.set_yticks(y); ax.grid(axis="x", zorder=0)
axes[0].set_yticklabels(names, fontsize=7.4)
S.save(fig, "F01_zeroshot_benchmarks")
