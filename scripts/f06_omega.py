#!/usr/bin/env python3
"""F06 / manuscript Figure 11 — angular-velocity (omega) ablation, seed 42.

(a) Training episodes spent in each curriculum phase, with and without omega.
(b) Benchmark B (dynamic obstacles) success and collision rate.

DATA: SEED 42 throughout, deliberately. The No-omega ablation was trained at seed 42 only,
so pooling three seeds for the baselines against one for the ablation would bias the
contrast. Seed-matched is the fair comparison and both panel titles say so.
NEEDS NOTHING FROM THE EVALUATION QUEUE — every input is already complete.

Reproduces the published values exactly: 1,440 vs 175 episodes in Phase 1;
13.8/86.2, 74.0/26.0, 77.0/23.0 on Benchmark B.

LABELS: (a) labels the two values the caption cites (175 and 1,440), as published.
        (b) labels every bar, as published."""
import sys, os, csv, collections; sys.path.insert(0, os.path.dirname(__file__))
import pv_style as S, pv_data as D
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import Patch
S.apply()

ph_no = collections.Counter()
for r in csv.DictReader(open(os.path.expanduser(
        "~/tb3_drl_logs/phase3/sac_pv_stam_no_omega_s42.csv"))):
    try: ph_no[int(r["phase"])] += 1
    except (KeyError, ValueError, TypeError): pass
ph_full = collections.Counter(D.train_log("v8", 42)[2].tolist())

fig, (a0, a1) = plt.subplots(1, 2, figsize=(S.FULL_W, 2.9),
                             gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.245})
fig.subplots_adjust(left=0.075, right=0.985, top=0.885, bottom=0.235)

# ── (a) episodes per curriculum phase ───────────────────────────────────────
PH = list(range(1, 8)); x = np.arange(len(PH)); w = 0.38
a0.bar(x-w/2, [ph_full.get(p, 0) for p in PH], w, color=S.GREEN,
       edgecolor=S.INK, lw=0.6, zorder=3)
a0.bar(x+w/2, [ph_no.get(p, 0) for p in PH], w, color=S.ORANGE,
       edgecolor=S.INK, lw=0.6, zorder=3)
top = max(max(ph_full.values()), max(ph_no.values()))
a0.text(x[0]-w/2, ph_full[1]+top*0.030, f"{ph_full[1]}", ha="center", fontsize=6.7)
a0.text(x[0]+w/2, ph_no[1]+top*0.030, f"{ph_no[1]:,}", ha="center", fontsize=6.7,
        fontweight="bold")
a0.set_ylim(0, top*1.14)
a0.set_xticks(x); a0.set_xticklabels([f"P{p}" for p in PH], fontsize=7.5)
a0.set_xlabel("Curriculum phase", fontsize=7.5)
a0.set_ylabel("Episodes in phase", fontsize=7.5)
a0.set_title("(a) Episodes per curriculum phase (seed 42)", fontsize=8.0)
a0.grid(axis="y", zorder=0)

# ── (b) Benchmark B outcome ─────────────────────────────────────────────────
rows = [("No-ω", S.ORANGE, "no_omega"),
        ("SAC-PV-STAM\n(with ω)", S.GREEN, "v8"),
        ("SAC-R-PV-STAM\n(with ω)", S.RED, "v11")]
x2 = np.arange(len(rows)); w2 = 0.30
for i, (lab, col, st) in enumerate(rows):
    r = D.eval_rows(st, 42, "benchmark_dqn_stage4")
    if not r: continue
    n = len(r)
    sr = sum(v["goal_reached"] for v in r)/n*100
    cr = sum(v["collision"] for v in r)/n*100
    a1.bar(x2[i]-w2/2, sr, w2, color=col, edgecolor=S.INK, lw=0.6, zorder=3)
    a1.bar(x2[i]+w2/2, cr, w2, color=col, alpha=0.45, hatch="//",
           edgecolor=S.INK, lw=0.6, zorder=3)
    a1.text(x2[i]-w2/2, sr+2.0, f"{sr:.1f}", ha="center", fontsize=6.7)
    a1.text(x2[i]+w2/2, cr+2.0, f"{cr:.1f}", ha="center", fontsize=6.7)
a1.set_xticks(x2); a1.set_xticklabels([r[0] for r in rows], fontsize=7.0)
a1.set_ylabel("Rate (%)", fontsize=7.5); a1.set_ylim(0, 100)
a1.set_title("(b) Benchmark B — dynamic (seed 42)", fontsize=8.0)
a1.grid(axis="y", zorder=0)
a1.legend(handles=[Patch(facecolor="white", edgecolor=S.INK, label="Success rate (SR)"),
                   Patch(facecolor="white", edgecolor=S.INK, hatch="//",
                         label="Collision rate (CR)")],
          loc="upper right", frameon=True, fontsize=6.6, framealpha=1.0,
          edgecolor="#CCCCCC", handlelength=1.5)

S.legend_below(fig, [Patch(facecolor=S.GREEN, edgecolor="none", label="SAC-PV-STAM (with ω)"),
                     Patch(facecolor=S.ORANGE, edgecolor="none", label="No-ω"),
                     Patch(facecolor=S.RED, edgecolor="none", label="SAC-R-PV-STAM (with ω)")],
              y=0.012, ncol=3, fontsize=7.4)
S.save(fig, "F06_omega_ablation")
