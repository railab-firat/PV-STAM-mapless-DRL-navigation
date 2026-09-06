#!/usr/bin/env python3
"""F10 / manuscript Figure 9 — PV-STAM attention-weight distributions.

WIDE LANDSCAPE BUILD, 600 dpi. Authored at S.LAND_W (9.6") with every font pre-scaled
by S.LAND_K, so the figure stays >= 7 pt even if Word places it at 6.3" full text width,
and is crisper still if given more room. Export is 600 dpi for detailed on-screen reading.

DATA: REAL logged observations, v8 s777 replay buffer (200,000 transitions), v8 s42 weights.
REDUCTION (stated, because it is a choice): attention received by each sector, averaged over
both heads and all 24 queries — the most conservative of the reductions tested.

🔴 THE PUBLISHED CAPTION IS CONTRADICTED BY THE DATA: it says "(a) Clear path -> near-uniform
attention", but Clear Path has the HIGHEST peak of all four panels under every reduction
(3.5x mean-reduced, 7.1x max-reduced). Panel (e) replaces the anecdote with a measurement."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import pv_style as S
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from matplotlib.lines import Line2D
S.apply()
K = S.LAND_K                              # font pre-scale factor
def F(pt): return round(pt*K, 1)

Z = np.load(os.path.expanduser("~/Desktop/PVSTAM_FIGURES_FINAL/data/attention_real.npz"))
order = [str(x) for x in Z["_order"]]
UNI, CLOSE = 1/24, 0.30/3.5
vmax = max(float(Z[k+"|attn"].max()) for k in order)
norm = mpl.colors.Normalize(0, vmax); cmap = plt.cm.YlOrRd
th = np.linspace(0, 2*np.pi, 24, endpoint=False); wid = 2*np.pi/24*0.86

fig = plt.figure(figsize=(S.LAND_W, 6.2))
gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 0.92], wspace=0.26, hspace=0.42,
                      left=0.045, right=0.885, top=0.905, bottom=0.135)

for i, k in enumerate(order):
    ax = fig.add_subplot(gs[0, i], projection="polar")
    w = Z[k+"|attn"]; sc = Z[k+"|scan"]; ds = Z[k+"|ds"]
    R = vmax*1.14
    for r in (UNI, vmax*0.5, vmax):
        ax.plot(np.linspace(0, 2*np.pi, 240), [r]*240, color="#E6E6E6", lw=0.7, zorder=1)
    ax.bar(th, w, width=wid, color=cmap(norm(w)), edgecolor="#6E6E6E", lw=0.45, zorder=3)
    ax.plot(np.append(th, th[0]), [UNI]*25, color="#3A3A3A", ls="--", lw=1.3, zorder=4)
    for j in range(24):
        if sc[j] < CLOSE:
            ax.plot([th[j]], [R*0.95], marker="o", ms=6.0,
                    color=S.BLUE if ds[j] < 0 else "#9A9A9A",
                    markeredgecolor="white", markeredgewidth=0.7, zorder=6)
    ax.plot([0], [0], marker="o", ms=6.2, color=S.GREEN,
            markeredgecolor="white", markeredgewidth=0.7, zorder=7)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.set_ylim(0, R)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax.set_xticklabels(["F", "L", "B", "R"], fontsize=F(7.2))
    ax.set_yticklabels([]); ax.grid(color="#F0F0F0", lw=0.5)
    ax.set_title(f"({chr(97+i)}) {k}\npeak {w.max()/UNI:.1f}× uniform",
                 fontsize=F(7.8), pad=9)

cax = fig.add_axes([0.900, 0.545, 0.0105, 0.335])
cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
cb.set_label("Attention weight", fontsize=F(7.4), labelpad=5)
cb.set_ticks([0, UNI, vmax/2, vmax])
cb.set_ticklabels(["0", f"{UNI:.3f} (uniform)", f"{vmax/2:.3f}", f"{vmax:.3f}"])
cb.ax.tick_params(labelsize=F(6.4))

ax = fig.add_subplot(gs[1, :])
al = Z["_align"]; ra = Z["_ratio"]
ax.hist(al, bins=np.arange(0, 14)-0.5, color=S.GREEN, edgecolor=S.INK, lw=0.7,
        density=True, zorder=3, label="observed")
ax.axhline(1/13, color=S.RED, ls="--", lw=1.6, zorder=4, label="chance (uniform)")
ax.set_xlabel("Sectors between peak attention and the nearest obstacle", fontsize=F(7.8))
ax.set_ylabel("Density", fontsize=F(7.8))
ax.tick_params(labelsize=F(7.0)); ax.grid(axis="y", zorder=0); ax.set_xlim(-0.7, 12.7)
ax.set_title(f"(e) Population measurement — n = {len(al):,} logged observations, "
             f"mean peak {ra.mean():.2f}× uniform", fontsize=F(7.8), pad=7)
ax.legend(frameon=False, fontsize=F(7.2), loc="upper right", bbox_to_anchor=(1.0, 1.03))
ax.text(0.985, 0.58, f"peak within 1 sector of the obstacle: {(al<=1).mean()*100:.1f}%"
        f"    chance: 12.5%", transform=ax.transAxes, ha="right", fontsize=F(7.2),
        color="#333333")

fig.legend(handles=[
    Line2D([],[],marker="o",color="none",markerfacecolor=S.BLUE,markersize=6.5,
           markeredgecolor="white", label="Approaching obstacle (Δs < 0, d < 0.30 m)"),
    Line2D([],[],marker="o",color="none",markerfacecolor="#9A9A9A",markersize=6.5,
           markeredgecolor="white", label="Static close obstacle"),
    Line2D([],[],marker="o",color="none",markerfacecolor=S.GREEN,markersize=6.5,
           markeredgecolor="white", label="Robot (R)"),
    Line2D([],[],color="#3A3A3A",ls="--",lw=1.4,label="Uniform attention (1/24)")],
    loc="lower center", bbox_to_anchor=(0.47, 0.008), ncol=4, frameon=True,
    fontsize=F(7.2), edgecolor="#CCCCCC", framealpha=1.0, handlelength=1.8,
    columnspacing=2.0)

out = os.path.expanduser("~/Desktop/PVSTAM_FIGURES_FINAL/figures_v2")
for ext, dpi in (("png", 600), ("pdf", 600), ("svg", 600)):
    fig.savefig(os.path.join(out, f"F10_attention_polar.{ext}"), dpi=dpi)
plt.close(fig)
print("  saved F10_attention_polar  (9.6in wide, 600 dpi)")
