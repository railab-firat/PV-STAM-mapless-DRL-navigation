#!/usr/bin/env python3
"""F09 / manuscript Figure 2 — scan-difference channel and rotation contamination.

STYLING: replicates the published figure — bold corner panel letters, the down-arrow
between rows, two-line titles with a bold "Mean spikes" line, left-aligned caveat banner.
LAYOUT: 3 rows x 2 cols. Rows 1-2 are the published (a)-(d); row 3 adds the hardware
measurement at identical thresholds.
DATA: (a)-(d) v8 s777 replay buffer, 200,000 transitions. (e)-(f) all 130 hardware trials.
NOTE: the published panel (a) shows a flat scan (~0.72, low variance). NO such frame exists
in the buffer — scan means top out at 0.61 and 0 of 11,421 sampled frames match that
profile. The frames used here are the flattest REAL ones available."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import pv_style as S
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch
S.apply()
repo_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "hardware"))
external_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "PVSTAM_PAPER_RESOURCES_ARCHIVE", "03_evaluation_datasets_csv"))
data_dir = repo_data if os.path.exists(os.path.join(repo_data, "fig2abcd.npz")) else external_data
A = np.load(os.path.join(data_dir, "fig2abcd.npz"))
H = np.load(os.path.join(data_dir, "fig2ef.npz"))
BLUE, RED, DARKRED, GREEN = S.BLUE, S.RED, S.RED_DK, S.GREEN
sec = np.arange(24); THR = -0.1

fig = plt.figure(figsize=(S.FULL_W, 7.4))
gs = fig.add_gridspec(3, 2, hspace=0.76, wspace=0.28,
                      left=0.085, right=0.965, top=0.88, bottom=0.08)
ax = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)])

# ── caveat banner, PERFECTLY CENTERED at top ─────────────────────────────────
fig.add_artist(FancyBboxPatch((0.085, 0.928), 0.88, 0.045,
    boxstyle="round,pad=0.008,rounding_size=0.012", transform=fig.transFigure,
    facecolor="#FDF6D8", edgecolor="#C9B458", lw=1.0, zorder=5))
fig.text(0.525, 0.950, "Caveat: Δs$_i$ is ego-centric — sharp robot rotation generates "
         "spurious negative values", ha="center", va="center", fontsize=7.2, zorder=6)

def corner(a, letter):
    """Bold panel letter at top-left, cleanly offset above the plot frame."""
    a.text(-0.14, 1.16, f"({letter})", transform=a.transAxes, fontsize=8.8,
           fontweight="bold", va="top", ha="left")

def title2(a, line1, line2):
    a.set_title(f"{line1}\n", fontsize=7.6, pad=10)
    a.text(0.5, 1.025, line2, transform=a.transAxes, ha="center", va="bottom",
           fontsize=7.2, fontweight="bold")

def scanpanel(a, v, colour, letter, title, mark=None):
    a.bar(sec, v, color=colour, edgecolor=S.INK, lw=0.35, zorder=3)
    if mark is not None:
        for j in np.where(mark)[0]:
            a.bar([j], [v[j]], color=RED, edgecolor=DARKRED, lw=0.9, hatch="///", zorder=4)
    a.set_ylim(0, 1.0); a.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    a.set_ylabel("Normalized distance", fontsize=7.0)
    a.set_title(title, fontsize=7.7, pad=10); corner(a, letter)

def dspanel(a, ds, letter, l1, l2):
    a.bar(sec, [d if d >= 0 else 0 for d in ds], color=BLUE, edgecolor=S.INK, lw=0.35, zorder=3)
    a.bar(sec, [d if d < 0 else 0 for d in ds], color=RED, edgecolor=S.INK, lw=0.35, zorder=3)
    for j in np.where(ds < THR)[0]:
        a.bar([j], [ds[j]], color=RED, edgecolor=DARKRED, lw=0.9, hatch="///", zorder=4)
    a.axhline(0, color=S.INK, lw=0.8, zorder=4)
    lim = max(0.28, float(np.abs(ds).max())*1.15)
    a.set_ylim(-lim, lim); a.set_ylabel("Δs$_i$ (normalized)", fontsize=7.0)
    title2(a, l1, l2); corner(a, letter)

def hwpanel(a, prof, letter, l1, l2, ymax):
    a.bar(sec, prof*100, color=RED, edgecolor=DARKRED, lw=0.5, hatch="///", zorder=3)
    a.set_ylim(0, ymax); a.set_ylabel("Sectors flagged (%)", fontsize=7.0)
    title2(a, l1, l2); corner(a, letter)

scanpanel(ax[0,0], A["lo_prev"], BLUE,  "a", "$s_i(t\\!-\\!1)$")
scanpanel(ax[0,1], A["lo_cur"],  GREEN, "b", "$s_i(t)$", mark=(A["lo_ds"] < THR))
dspanel(ax[1,0], A["lo_ds"], "c", "Δs$_i$   ($|\\omega| \\leq 0.1$ rad/s)",
        f"Mean spikes < −0.1:  {float(A['lo_rate']):.2f}")
dspanel(ax[1,1], A["hi_ds"], "d", "Δs$_i$   ($|\\omega| > 1.0$ rad/s)",
        f"Mean spikes < −0.1:  {float(A['hi_rate']):.2f}")
_HW = max(float(H["lo_spike"].max()), float(H["hi_spike"].max()))*100*1.15
hwpanel(ax[2,0], H["lo_spike"], "e", "Hardware   ($|\\omega| \\leq 0.1$ rad/s)",
        f"Mean spikes < −0.1:  {float(H['lo_rate']):.2f}", _HW)
hwpanel(ax[2,1], H["hi_spike"], "f", "Hardware   ($|\\omega| > 1.0$ rad/s)",
        f"Mean spikes < −0.1:  {float(H['hi_rate']):.2f}", _HW)

for a in ax.ravel():
    a.set_xlabel("Sector index $i$", fontsize=7.0)
    a.set_xlim(-0.7, 23.7); a.set_xticks([0, 6, 12, 18, 23])
    a.grid(axis="y", zorder=0)

# down-arrow between the scan row and the Δs row, precisely aligned over column midpoints
for xf in (0.285, 0.745):
    fig.add_artist(FancyArrowPatch((xf, 0.655), (xf, 0.620), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=15, color=S.INK, lw=1.5, zorder=6))

fig.legend(handles=[
    Patch(facecolor=BLUE, edgecolor=S.INK, label="Δs$_i$ ≥ 0 — receding / static"),
    Patch(facecolor=RED, edgecolor=S.INK, label="Δs$_i$ < 0 — approaching obstacle"),
    Patch(facecolor=RED, edgecolor=DARKRED, hatch="///", label="Spurious Δs$_i$ < 0 — rotation artefact")],
    loc="lower center", bbox_to_anchor=(0.525, 0.005), ncol=3, frameon=False,
    fontsize=6.8, columnspacing=1.2, handlelength=1.4, handleheight=1.0)
S.save(fig, "F09_scan_difference_contamination")

