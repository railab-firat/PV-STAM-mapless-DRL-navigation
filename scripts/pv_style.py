"""
PV-STAM figure style — SINGLE SOURCE OF TRUTH for colour and layout.
Change a value here and every figure follows on the next run.

Palette matches the published manuscript (seaborn "deep" family).
Every red in every figure is COLLISION_RED. Every blue is BLUE. No exceptions.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as _pe

# ─────────────────────────────────────────────────────────────────────────────
# 1. THE PALETTE.  Nothing outside this block may define a colour.
# ─────────────────────────────────────────────────────────────────────────────
GREY    = "#8C8C8C"   # SAC-MLP / neutral baseline
BLUE    = "#4C72B0"   # SAC-MLP-FS / "receding or static"
GREEN   = "#55A868"   # SAC-PV-STAM / "goal reached"
GOLD    = "#CCB974"   # SAC-PV-STAM-H (384)
GOLD_LT = "#E8D9A8"   # SAC-PV-STAM-H (256) — same hue, lighter
RED     = "#C44E52"   # SAC-R-PV-STAM / "collision" / "approaching obstacle"
PURPLE  = "#8172B3"   # SAC-LSTM
ORANGE  = "#DD8452"   # ablated / alternative condition (No-omega)
SLATE   = "#5F7D95"   # single-series magnitude bars
LIGHT   = "#BFBFBF"   # "timeout" / neutral fill
RED_DK  = "#8A2F33"   # edge for hatched red (rotation artefact) — same hue as RED
INK     = "#333333"   # all outlines and axis furniture

V = {                                    # variant  ->  colour
    "SAC-MLP":             GREY,
    "SAC-MLP-FS":          BLUE,
    "SAC-PV-STAM":         GREEN,
    "SAC-PV-STAM-H (384)": GOLD,
    "SAC-PV-STAM-H (256)": GOLD_LT,
    "SAC-R-PV-STAM":       RED,
    "SAC-LSTM":            PURPLE,
}
ORDER = list(V)

OUT = {                                  # episode outcome -> colour (semantic, fixed)
    "goal":      GREEN,
    "timeout":   LIGHT,
    "collision": RED,
}
H = {k: "" for k in V}                   # no variant hatching

# Line style per variant — matches the published Figure 7 so curves stay
# distinguishable where they overlap, and survive greyscale printing.
LS = {
    "SAC-MLP":             (0, (5, 3)),      # dashed
    "SAC-MLP-FS":          (0, (6, 2, 1, 2)),# dash-dot
    "SAC-PV-STAM":         "solid",
    "SAC-PV-STAM-H (384)": (0, (1, 2)),      # dotted
    "SAC-PV-STAM-H (256)": (0, (1, 2)),
    "SAC-R-PV-STAM":       "solid",
    "SAC-LSTM":            (0, (4, 2, 1, 2)),
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. LABELLING POLICY  (applied identically in every figure)
#    - stacked composition bars  -> EVERY segment labelled
#    - magnitude bars            -> EVERY bar labelled, or none; never a subset
# ─────────────────────────────────────────────────────────────────────────────
LABEL_INSIDE_MIN = 9.0    # % below which a stacked label is moved outside the bar

# ─────────────────────────────────────────────────────────────────────────────
# PAGE GEOMETRY — author figures AT print size, never larger.
# A figure drawn 12" wide and placed at 6.3" has every font halved on the page.
# MDPI: single column 3.35", full text width 6.3". Keep 9 pt type at 1.0 scale.
# ─────────────────────────────────────────────────────────────────────────────
COL_W  = 3.35    # single column
FULL_W = 6.30    # full text width  <- use this for every multi-panel figure
WIDE_W = 7.00    # only if the figure will be placed in landscape / rotated
LAND_W = 9.60    # full landscape width. Fonts MUST be pre-scaled by LAND_W/FULL_W
                 # so they still clear 7 pt if the figure is placed at 6.3".
LAND_K = LAND_W / FULL_W          # = 1.52 - multiply every font size by this


def apply():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Tinos", "Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8.6, "axes.labelsize": 8.2,
        "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
        "legend.fontsize": 7.6,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK,
        "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
    })

def luminance(c):
    if isinstance(c, str):
        c = c.lstrip("#"); c = tuple(int(c[i:i+2], 16)/255 for i in (0, 2, 4))
    r, g, b = [(v/12.92 if v <= 0.04045 else ((v+0.055)/1.055)**2.4) for v in c]
    return 0.2126*r + 0.7152*g + 0.0722*b

def on_color(fill):
    """Ink that stays legible on `fill`."""
    return "#FFFFFF" if luminance(fill) < 0.45 else "#1A1A1A"

def err_kw(lw=1.3, capsize=3.0):
    return dict(ecolor="#1A1A1A", elinewidth=lw, capsize=capsize, capthick=lw)

def style_errorbars(container, halo=3.2):
    """White halo so error bars read INSIDE a dark bar and OUTSIDE against the page."""
    eb = getattr(container, "errorbar", None)
    if eb is None:
        return container
    fx = [_pe.withStroke(linewidth=halo, foreground="white")]
    for grp in eb.lines:
        if grp is None:
            continue
        for it in (grp if isinstance(grp, (tuple, list)) else [grp]):
            try: it.set_path_effects(fx)
            except AttributeError: pass
    return container

def legend_below(fig, handles, y=0.010, ncol=None, fontsize=9.2):
    """The only legend style used anywhere: horizontal, centred, below the axes."""
    lg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y),
                    ncol=ncol or len(handles), frameon=False, fontsize=fontsize,
                    columnspacing=2.2, handlelength=1.6, handleheight=1.15,
                    handletextpad=0.6)
    for t in lg.get_texts():
        t.set_fontweight("semibold")
    return lg

def variant_handles(names):
    from matplotlib.patches import Patch
    return [Patch(facecolor=V[n], edgecolor="none", label=n) for n in names]

def outcome_handles():
    from matplotlib.patches import Patch
    return [Patch(facecolor=OUT["goal"], edgecolor="none", label="Goal reached"),
            Patch(facecolor=OUT["timeout"], edgecolor="none", label="Timeout"),
            Patch(facecolor=OUT["collision"], edgecolor="none", label="Collision")]

def stacked_label(ax, val, centre, row, horizontal=True, fill=None):
    """Label EVERY segment. Small ones move outside so none is silently dropped."""
    if val <= 0:
        return
    txt = f"{val:.1f}"
    if val >= LABEL_INSIDE_MIN:
        xy = (centre, row) if horizontal else (row, centre)
        ax.text(*xy, txt, ha="center", va="center", fontsize=8.6,
                color=on_color(fill), fontweight="bold", zorder=6)
    else:
        xy = (centre, row + 0.42) if horizontal else (row, centre)
        ax.text(*xy, txt, ha="center", va="bottom", fontsize=7.8,
                color="#1A1A1A", fontweight="bold", zorder=6)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SCHEMATIC FILLS — tints of the SAME hues used in the data figures, so the
#    diagrams and the plots read as one set. Role -> meaning is fixed:
#       input      sensor data entering the system
#       process    deterministic transform (no learned attention)
#       novel      THE CONTRIBUTION - attention / dual channel. Always highlighted.
#       policy     learned actor-critic
#       perturb    deliberately injected delay or noise
#       terminal   final / hardest / output stage
# ─────────────────────────────────────────────────────────────────────────────
FILL = {
    "input":    "#DCE6F1",   # tint of BLUE
    "process":  "#DDEBE0",   # tint of GREEN
    "novel":    "#F5EBD0",   # tint of GOLD   <- the contribution
    "policy":   "#DEE5EA",   # tint of SLATE
    "perturb":  "#F5DEDF",   # tint of RED
    "terminal": "#CFE3D6",   # deeper GREEN tint
}
EDGE = {
    "input": BLUE, "process": GREEN, "novel": GOLD,
    "policy": SLATE, "perturb": RED, "terminal": GREEN,
}

def block(ax, x, y, w, h, title, sub=None, role="process", highlight=False,
          fs=9.0, subfs=7.4):
    """One layer of a schematic. `highlight=True` marks the novel contribution."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.30,rounding_size=0.9",
        facecolor=FILL[role], edgecolor=EDGE[role],
        linewidth=2.2 if highlight else 1.1, zorder=3))
    ax.text(x+w/2, y+h/2+(h*0.16 if sub else 0), title, ha="center", va="center",
            fontsize=fs, fontweight="bold" if highlight else "normal", zorder=4)
    if sub:
        ax.text(x+w/2, y+h/2-h*0.22, sub, ha="center", va="center",
                fontsize=subfs, color="#555555", zorder=4)

def arrow(ax, x1, y1, x2, y2, label=None, dashed=False, curve=0.0):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
        mutation_scale=11, color=INK, linewidth=1.1,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={curve}" if curve else "arc3", zorder=2))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2, label, ha="center", va="bottom",
                fontsize=7.2, color="#444444", zorder=4,
                bbox=dict(fc="white", ec="none", pad=1.2))

def note(ax, x, y, text, fs=7.4):
    ax.text(x, y, text, ha="left", va="center", fontsize=fs, color="#666666",
            style="italic", zorder=4)

def save(fig, name, outdir=None):
    import os
    outdir = outdir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "01_final_rendered_figures_png_pdf_svg"))
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  saved {name} to {outdir}")
