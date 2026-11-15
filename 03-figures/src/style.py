"""Shared figure style for *Transformers from First Principles* (Math & CS from First Principles).
Palette (series manual section 9 as restated in the book prompt): navy = learned components,
amber = data, plum = values, probabilities and cost, slate blue = cached state, teal = tools,
coral red = breakages. Blue is the accent for structure and highlights."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# Semantic palette - every figure uses these roles consistently
CORAL  = "#e4573d"   # breakages
SLATEBLUE = "#5b7aa6" # cached state
NAVY   = "#1f3f66"   # structure, ink, headings
BLUE   = "#2a6fb0"   # the model / reasoning
TEAL   = "#1b9e8a"   # tools / actions
PLUM   = "#7b4fa0"   # memory / state
AMBER  = "#c98f12"   # context / data
VERM   = "#d8572a"   # risk / failure / cost
GREEN  = "#3b8f3b"   # humans
SLATE  = "#3c4a5a"   # body text in figures
GREY   = "#6b7a8f"   # secondary labels
LIGHT  = "#e8ecf2"   # panels
PAPER  = "#ffffff"
TINT = {BLUE: "#e3eef8", TEAL: "#dff2ee", PLUM: "#efe7f5", AMBER: "#f8efd9",
        VERM: "#fbe6de", "#e4573d": "#fce4df", "#5b7aa6": "#e6ecf5", GREEN: "#e2f1e2", NAVY: "#e2e8f0", GREY: "#eef1f5"}
SERIES = [BLUE, VERM, TEAL, PLUM, AMBER, GREEN, NAVY]

def setup():
    plt.rcParams.update({
        "font.family": ["Inter", "DejaVu Sans"], "font.size": 11,
        "axes.edgecolor": "#9aa5b1", "axes.labelcolor": SLATE, "axes.titlecolor": NAVY,
        "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
        "xtick.color": SLATE, "ytick.color": SLATE, "axes.grid": True,
        "grid.color": "#e3e7ee", "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 100,
        "savefig.dpi": 200, "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "lines.linewidth": 2.2,
    })

def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12, facecolor=PAPER)
    plt.close(fig)

def canvas(w=9, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w * 10); ax.set_ylim(0, h * 10); ax.axis("off")
    return fig, ax

def box(ax, x, y, w, h, text, color=BLUE, fill=True, fs=11, weight="bold", tc=None, sub=None, r=1.6, lw=1.8, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=TINT.get(color, LIGHT) if fill else "white", ec=color, lw=lw, ls=ls)
    ax.add_patch(p)
    if sub:
        ax.text(x + w / 2, y + h / 2 + h * 0.14, text, ha="center", va="center", fontsize=fs, color=tc or color, weight=weight)
        ax.text(x + w / 2, y + h / 2 - h * 0.2, sub, ha="center", va="center", fontsize=fs - 2.2, color=SLATE)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc or color, weight=weight)
    return p

def arrow(ax, x1, y1, x2, y2, color=SLATE, lw=1.8, rad=0.0, style="-|>", ms=14, ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms, color=color,
                        lw=lw, connectionstyle=f"arc3,rad={rad}", linestyle=ls)
    ax.add_patch(a); return a

def label(ax, x, y, text, color=SLATE, fs=10, ha="center", va="center", **kw):
    ax.text(x, y, text, color=color, fontsize=fs, ha=ha, va=va, **kw)

def title(ax, text, x=0.5, y=None, fs=14):
    ax.text(x, (y if y is not None else ax.get_ylim()[1] - 1.5), text, fontsize=fs, weight="bold", color=NAVY, ha="left", va="top")
