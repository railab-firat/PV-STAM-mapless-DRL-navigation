#!/usr/bin/env python3
"""F05 / manuscript Figure 10 — Phase-7 navigation quality.

(a) Episode-outcome distribution, SEED 42, n=200 — the published basis. Reproduces the
    published values EXACTLY for 4 of 5 variants (MLP-FS 53.0, PV-STAM 66.0/6.5/27.5,
    PV-STAM-H 56.5/22.5/21.0, R-PV-STAM 68.5). SAC-MLP has no seed-42 file yet; a row
    falls back to another seed and is labelled with the seed actually used.
(b) Mean steps to goal over successful episodes, all COMPLETE seeds pooled.

Incomplete evaluations are excluded automatically (pv_data.MIN_EVAL_EPISODES), so a
run still in progress never leaks in. Re-run this script after eval_queue.sh finishes."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import pv_style as S, pv_data as D
import matplotlib.pyplot as plt, numpy as np
S.apply()

PUBLISHED = ["SAC-MLP", "SAC-MLP-FS", "SAC-PV-STAM", "SAC-PV-STAM-H (384)", "SAC-R-PV-STAM"]
V = [(st, nm) for st, nm in D.VARIANTS if nm in PUBLISHED
     and any(D.phase_eval(st, sd, 7) for sd in D.SEEDS)]
G, T, C = S.OUT["goal"], S.OUT["timeout"], S.OUT["collision"]

fig, (a0, a1) = plt.subplots(1, 2, figsize=(S.FULL_W, 2.9),
                             gridspec_kw={"width_ratios": [1.10, 1.0], "wspace": 0.07})
fig.subplots_adjust(left=0.155, right=0.982, top=0.885, bottom=0.205)
y = np.arange(len(V))[::-1]
labels = []

# ── (a) outcome distribution ────────────────────────────────────────────────
for i, (st, nm) in enumerate(V):
    rows, used = D.phase_eval(st, 42, 7), 42
    if not rows:
        used = next(sd for sd in D.SEEDS if D.phase_eval(st, sd, 7))
        rows = D.phase_eval(st, used, 7)
    labels.append(nm if used == 42 else f"{nm}\n(seed {used})")
    n = len(rows); g = sum(x["goal"] for x in rows); c = sum(x["coll"] for x in rows)
    left = 0.0
    for val, col in ((g/n*100, G), ((n-g-c)/n*100, T), (c/n*100, C)):
        if val <= 0: continue
        a0.barh(y[i], val, left=left, color=col, edgecolor="white", lw=0.8, zorder=3)
        a0.text(left+val/2, y[i], f"{val:.1f}", ha="center", va="center", fontsize=6.7,
                color=S.on_color(col), fontweight="bold", zorder=6)
        left += val
a0.set_xlim(0, 100); a0.set_xticks([0, 20, 40, 60, 80, 100])
a0.set_yticks(y); a0.set_yticklabels(labels, fontsize=7.0)
a0.set_xlabel("Episodes (%)", fontsize=7.5)
a0.set_title("(a) Phase-7 episode outcomes", fontsize=8.0)
a0.grid(axis="x", zorder=0)

# ── (b) navigation efficiency ───────────────────────────────────────────────
DAGGER = {"SAC-PV-STAM-H (384)"}      # low mean reflects selection bias, not efficiency
for i, (st, nm) in enumerate(V):
    v = np.array([x["steps"] for sd in D.SEEDS for x in D.phase_eval(st, sd, 7)
                  if x["goal"] > 0.5])
    if not len(v): continue
    bc = a1.barh(y[i], v.mean(), xerr=v.std(), color=S.SLATE, edgecolor=S.INK, lw=0.7,
                 zorder=3, error_kw=S.err_kw(lw=1.2, capsize=3.2))
    S.style_errorbars(bc, halo=2.2)
    a1.text(v.mean()+v.std()+5, y[i], f"{v.mean():.0f}±{v.std():.0f}"
            + (" †" if nm in DAGGER else ""), va="center", fontsize=6.7, color="#333333")
a1.set_yticks(y); a1.set_yticklabels([]); a1.set_xlim(0, 215)
a1.set_xticks([0, 50, 100, 150, 200])
a1.set_xlabel("Mean steps to goal   (1 step ≈ 150 ms at 6.67 Hz)", fontsize=7.2)
a1.set_title("(b) Navigation efficiency", fontsize=8.0)
a1.grid(axis="x", zorder=0)

S.legend_below(fig, S.outcome_handles(), y=0.012, ncol=3, fontsize=7.4)
S.save(fig, "F05_phase7_navigation_quality")
