#!/usr/bin/env python3
"""F08 — Real-robot episode outcomes, 130-trial cardboard-only campaign.
   Replaces the published Figure 14, which used 1-2 walking human participants.

THIS IS NOT A CORRECTION OF THE PUBLISHED PILOT — it is a different, larger experiment:
130 trials vs 40, cardboard obstacles instead of human participants, and a controlled
corridor instead of a randomized arena. The numbers are not expected to match and must
not be presented as if the pilot's values had simply been revised.

SCENARIOS ARE NEVER POOLED. The variant mix differs between them, so a pooled bar would be
a weighted average with variant-dependent weights.

WHY PANEL (b) HAS THREE VARIANTS, NOT FIVE: SAC-MLP and SAC-PV-STAM-H both scored 0% goal /
100% timeout in the EASIER static scenario, so the moving-obstacle condition was never run
for them. They are absent rather than drawn as zero, because no trials exist - a zero bar
would assert a measurement that was never made."""
import sys, os, csv; sys.path.insert(0, os.path.dirname(__file__))
import pv_style as S
import matplotlib.pyplot as plt, numpy as np
S.apply()

repo_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "hardware"))
external_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "PVSTAM_PAPER_RESOURCES_ARCHIVE", "04_hardware_trials_rosbags"))
trials_csv = os.path.join(repo_data, "trials_index.csv") if os.path.exists(os.path.join(repo_data, "trials_index.csv")) else os.path.join(external_data, "trials_index.csv")
idx = [r for r in csv.DictReader(open(trials_csv)) if r["decision"] == "saved"]
NAME = {"baseline": "SAC-MLP", "mlp_fs": "SAC-MLP-FS", "v8": "SAC-PV-STAM",
        "v10": "SAC-PV-STAM-H (384)", "v11": "SAC-R-PV-STAM"}
ORDER = ["SAC-MLP", "SAC-PV-STAM-H (384)", "SAC-PV-STAM", "SAC-MLP-FS", "SAC-R-PV-STAM"]
G, T, C = S.OUT["goal"], S.OUT["timeout"], S.OUT["collision"]

def cells(scen):
    out = []
    for nm in ORDER:
        key = next(k for k, v in NAME.items() if v == nm)
        rows = [r for r in idx if r["model"] == key and r["scenario"] == scen]
        if not rows: continue
        n = len(rows)
        g = sum(1 for r in rows if r["outcome"] == "goal")/n*100
        c = sum(1 for r in rows if r["outcome"] == "collision")/n*100
        out.append((nm, n, g, 100-g-c, c))
    return out

S2, S3 = cells("2"), cells("3")
fig, axes = plt.subplots(1, 2, figsize=(S.FULL_W, 3.2), sharey=True,
                         gridspec_kw={"width_ratios": [len(S2), len(S3)], "wspace": 0.07})
fig.subplots_adjust(left=0.078, right=0.985, top=0.885, bottom=0.245)

for ax, data, title in ((axes[0], S2, "(a) Static corridor"),
                        (axes[1], S3, "(b) Moving obstacle")):
    xs = np.arange(len(data))
    for i, (nm, n, g, t, c) in enumerate(data):
        bot = 0.0
        for val, col in ((g, G), (t, T), (c, C)):
            if val <= 0: continue
            ax.bar(i, val, bottom=bot, width=0.62, color=col,
                   edgecolor="white", lw=0.9, zorder=3)
            if val >= S.LABEL_INSIDE_MIN:
                ax.text(i, bot+val/2, f"{val:.0f}%", ha="center", va="center",
                        fontsize=6.9, color=S.on_color(col), fontweight="bold", zorder=6)
            else:
                ax.text(i+0.37, bot+val/2, f"{val:.0f}%", ha="left", va="center",
                        fontsize=6.2, color="#1A1A1A", fontweight="bold", zorder=6)
            bot += val
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{nm.replace(' (384)','')}\n(n={n})" for nm, n, *_ in data],
                       fontsize=6.7)
    ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.68, len(data)-0.32)
    ax.set_title(title, fontsize=8.0); ax.grid(axis="y", zorder=0)
axes[0].set_ylabel("Episode outcome (%)", fontsize=7.6)

S.legend_below(fig, S.outcome_handles(), y=0.012, ncol=3, fontsize=7.4)
S.save(fig, "F08_hardware_outcomes")
