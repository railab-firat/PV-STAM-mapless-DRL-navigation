#!/usr/bin/env python3
"""F04 / manuscript Figure 8 — success rate across curriculum Phases 5-7.

(a) Mean SR per phase per variant, error bars = ±1 SD over available seeds.
(b) Maximum curriculum phase reached, per seed.

VARIANTS: the 5 published ones. SAC-PV-STAM-H (256) and SAC-LSTM are omitted for now
because they have NO phase-level evaluation yet — panel (a) would be blank for them.
An evaluation queue is generating those 18 runs; regenerate with PUBLISHED_ONLY = False
once it finishes.

(b) METRIC: maximum phase EVER reached, matching the published panel title. This differs
from the final phase for runs that were demoted — SAC-PV-STAM seed 42 reached Phase 7 but
finished at Phase 6. Source is the training log where usable, else curriculum_state_*.json
for the four runs whose logs are fragments.

LABELS: only bars that fall short of Phase 7 are labelled, as published — bars at the
target need no number because the dashed target line marks it."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import pv_style as S, pv_data as D
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import Patch
S.apply()

PUBLISHED_ONLY = False
PUBLISHED = ["SAC-MLP", "SAC-MLP-FS", "SAC-PV-STAM", "SAC-PV-STAM-H (384)", "SAC-R-PV-STAM"]
V = [(st, nm) for st, nm in D.VARIANTS if (nm in PUBLISHED or not PUBLISHED_ONLY)]

fig, (a0, a1) = plt.subplots(1, 2, figsize=(S.FULL_W, 3.1),
                             gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.235})
fig.subplots_adjust(left=0.072, right=0.985, top=0.905, bottom=0.245)

# ── (a) mean SR per phase ───────────────────────────────────────────────────
PH = [5, 6, 7]; x = np.arange(len(PH)); w = 0.80/len(V)
for i, (st, nm) in enumerate(V):
    xs, mu, sd_ = [], [], []
    for j, p in enumerate(PH):
        vals = [g/n*100 for s in D.SEEDS for n, g in [D.phase_sr(st, s, p)] if n > 0]
        if not vals: continue
        xs.append(x[j] + (i-(len(V)-1)/2)*w); mu.append(np.mean(vals)); sd_.append(np.std(vals))
    if not mu: continue
    bc = a0.bar(xs, mu, w*0.90, yerr=sd_, color=S.V[nm], edgecolor=S.INK, lw=0.7, zorder=3,
                error_kw=S.err_kw(lw=1.3, capsize=2.8))
    S.style_errorbars(bc)
a0.set_xticks(x); a0.set_xticklabels([f"Phase {p}" for p in PH], fontsize=7.6)
a0.set_ylabel("Mean success rate (%)", fontsize=7.6)
a0.set_ylim(0, 100); a0.grid(axis="y", zorder=0)
a0.set_title("(a)", fontsize=8.4, loc="left", fontweight="bold")

# ── (b) max curriculum phase reached, per seed ──────────────────────────────
HATCH = {42: "", 777: "//", 123: "\\\\"}
x2 = np.arange(len(V)); w2 = 0.26
for i, (st, nm) in enumerate(V):
    for k, sd in enumerate(D.SEEDS):
        ph, src = D.max_phase(st, sd)
        if ph is None: continue
        a1.bar(x2[i]+(k-1)*w2, ph, w2*0.82, color=S.V[nm], hatch=HATCH[sd],
               edgecolor=S.INK, lw=0.7, zorder=3)
        if ph < 7:                       # label only runs that missed the target
            a1.text(x2[i]+(k-1)*w2, ph+0.14, str(ph), ha="center", fontsize=6.7)
a1.axhline(7, color=S.GOLD, ls="--", lw=1.4, zorder=4)
a1.text(len(V)-0.42, 7.18, "Ph. 7 target", fontsize=6.9, color="#A8822F", ha="right")
a1.set_xticks(x2)
a1.set_xticklabels([n.replace("SAC-", "").replace(" (384)", "") for _, n in V],
                   rotation=22, ha="right", fontsize=6.9)
a1.set_yticks(range(0, 8)); a1.set_ylim(0, 9.0)
a1.set_ylabel("Max curriculum phase reached", fontsize=7.6)
a1.grid(axis="y", zorder=0)
a1.set_title("(b)", fontsize=8.4, loc="left", fontweight="bold")
a1.legend(handles=[Patch(facecolor="white", hatch=HATCH[s], edgecolor=S.INK, label=f"Seed {s}")
                   for s in D.SEEDS], frameon=False, fontsize=6.7, loc="upper left",
          bbox_to_anchor=(0.005, 0.99), ncol=3, handlelength=1.3, columnspacing=0.9)

S.legend_below(fig, S.variant_handles([n for _, n in V]), y=0.012, ncol=len(V), fontsize=7.2)
S.save(fig, "F04_phase_progression")
