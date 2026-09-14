#!/usr/bin/env python3
"""Fig 13 — failure modes, five architectures, static corridor trials.
Top: path coloured by elapsed time, goal disc shaded. Bottom: progress vs time."""
import sys, os, csv
sys.path.insert(0, os.path.dirname(__file__))
import pv_style as S
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
S.apply()

repo_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "hardware"))
external_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "PVSTAM_PAPER_RESOURCES_ARCHIVE"))
bags_p = os.path.join(repo_data, "bags.npz") if os.path.exists(os.path.join(repo_data, "bags.npz")) else os.path.join(external_data, "03_evaluation_datasets_csv", "bags.npz")
trials_p = os.path.join(repo_data, "trials_index.csv") if os.path.exists(os.path.join(repo_data, "trials_index.csv")) else os.path.join(external_data, "04_hardware_trials_rosbags", "trials_index.csv")
d = np.load(bags_p)
idx = {r["trial_id"]: r for r in csv.DictReader(open(trials_p))}

# scenario-2 (static) trials, one column per architecture
COLS = [("baseline_s777_scen2", "SAC-MLP"),
        ("v10_s777_scen2",      "SAC-PV-STAM-H (384)"),
        ("v8_s777_scen2",       "SAC-PV-STAM"),
        ("mlp_fs_s777_scen2",   "SAC-MLP-FS"),
        ("v11_s777_scen2",      "SAC-R-PV-STAM")]
GOAL, GOAL_R = 4.5, 0.40

fig, axes = plt.subplots(2, 5, figsize=(S.FULL_W, 3.4),
                         gridspec_kw={"height_ratios": [0.78, 1.0], "hspace": 0.55, "wspace": 0.28})

for c, (prefix, label) in enumerate(COLS):
    tids = sorted({k.split("|")[0] for k in d.files if k.startswith(prefix)})
    axT, axB = axes[0, c], axes[1, c]
    col = S.V[label]

    # goal region
    axT.add_patch(plt.Circle((GOAL, 0), GOAL_R, color=S.GREEN, alpha=0.16, zorder=1))
    axT.axvline(GOAL, color=S.GREEN, ls=":", lw=0.9, zorder=2)
    finals = []
    for t in tids:
        ox, oy, ot = d[f"{t}|ox"], d[f"{t}|oy"], d[f"{t}|ot"]
        x = ox - ox[0]; y = oy - oy[0]
        pts = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis",
                            norm=plt.Normalize(0, ot[-1]), lw=1.0, alpha=0.85, zorder=3)
        lc.set_array(ot[:-1]); axT.add_collection(lc)
        prog = np.hypot(x, y)
        axB.plot(ot, prog, color=col, lw=0.9, alpha=0.6, zorder=3)
        finals.append(np.maximum.accumulate(prog)[-1])

    axT.set_xlim(-0.4, 5.2); axT.set_ylim(-1.65, 1.65); axT.set_aspect("equal")
    axT.set_title(label, fontsize=7.2, color=col, pad=6)
    axT.grid(zorder=0); axT.set_xlabel("x (m)", fontsize=6.4)
    if c == 0: axT.set_ylabel("y (m)", fontsize=6.4)

    axB.axhline(GOAL, color=S.GREEN, ls=":", lw=0.9, zorder=2)
    axB.set_xlim(0, 160); axB.set_ylim(0, 5.2); axB.grid(zorder=0)
    axB.set_xlabel("Time (s)", fontsize=6.4)
    if c == 0: axB.set_ylabel("Distance travelled (m)", fontsize=6.4)
    axB.text(0.95, 0.45, f"max {np.max(finals):.2f} m\nmed {np.median(finals):.2f} m",
             transform=axB.transAxes, ha="right", va="center", fontsize=5.4, color="#333333",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#D8D8D8", alpha=0.85, lw=0.6), zorder=5)

sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
cb = fig.colorbar(sm, ax=axes[0, :].tolist(), fraction=0.010, pad=0.010, aspect=12)
cb.set_label("Elapsed time (normalised)", fontsize=6.0); cb.ax.tick_params(labelsize=7)

fig.subplots_adjust(bottom=0.20)
S.save(fig, "F11_hardware_failure_modes")
