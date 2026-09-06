#!/usr/bin/env python3
"""F07 / manuscript Figure 7 — rolling success rate, Phase-7 deterministic evaluation.

DATA: sac_{variant}_s{seed}_ph7_eval.csv, 20-episode rolling window, mean over ALL
      available seeds. Seed count is printed beside each variant in the legend.
      Coverage: MLP 1 seed (777) · MLP-FS 3 · PV-STAM 3 · PV-STAM-H 2 · R-PV-STAM 2.

NO SHADED BANDS - deliberate, and see the doc: the published text claims SAC-MLP shows
"the widest shaded band, reflecting substantial inter-seed variance", but SAC-MLP has only
ONE Phase-7 seed, so no such band can exist. That sentence needs revision.

STYLE: per-variant line styles replicate the published figure (dashed / dash-dot / solid /
dotted) so overlapping curves stay separable and survive greyscale printing."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import pv_style as S, pv_data as D
import matplotlib.pyplot as plt, numpy as np
from matplotlib.lines import Line2D
S.apply()

fig, ax = plt.subplots(figsize=(S.FULL_W, 3.2))
fig.subplots_adjust(left=0.085, right=0.905, top=0.94, bottom=0.255)

handles = []
for st, nm in D.VARIANTS:
    runs = [D.rolling([v["goal"] for v in D.phase_eval(st, sd, 7)], 20)[1]
            for sd in D.SEEDS if D.phase_eval(st, sd, 7)]
    if not runs:
        continue
    L = min(len(r) for r in runs)
    ax.plot(np.arange(1, L+1), np.array([r[:L] for r in runs]).mean(0),
            color=S.V[nm], lw=1.7, ls=S.LS[nm], zorder=3,
            solid_capstyle="round", dash_capstyle="round")
    handles.append(Line2D([], [], color=S.V[nm], lw=1.9, ls=S.LS[nm],
                          label=f"{nm}  ({len(runs)} seed{'s' if len(runs) > 1 else ''})"))

ax.axhline(50, color="#8A8A8A", ls=(0, (6, 4)), lw=1.1, zorder=2)
ax.text(1.010, 0.50, "50%\nreference", transform=ax.transAxes, va="center", ha="left",
        fontsize=6.9, color="#8A8A8A", linespacing=1.35)

ax.set_xlabel("Evaluation episode", fontsize=7.6)
ax.set_ylabel("Rolling success rate, window = 20 (%)", fontsize=7.6)
ax.set_ylim(0, 100); ax.set_xlim(1, 200)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.grid(zorder=0)

lg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
                ncol=3, frameon=False, fontsize=7.0, columnspacing=2.4,
                handlelength=2.6, handletextpad=0.7, labelspacing=0.7)
for t in lg.get_texts():
    t.set_fontweight("semibold")
S.save(fig, "F07_rolling_sr_phase7")
