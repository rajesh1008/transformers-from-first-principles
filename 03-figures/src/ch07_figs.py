"""Figures for Chapter 7 (Many at Once), computed with the same code as the listings.
Run from the project root: python3 03-figures/src/ch07_figs.py"""
import os, sys, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code"))
from style import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from tiny.tokenizer import TOY_VOCAB
from tiny.embed import TOY_EMBED, TOY_TURN, TOY_POS, apply_rotary
from tiny.linalg import intensity
setup()
OUT = os.path.join(ROOT, "03-figures", "ch07"); os.makedirs(OUT, exist_ok=True)
REC = os.path.join(ROOT, "04-code", "ch07", "outputs")
P = lambda n: os.path.join(OUT, n)
SENT = "the cat sat on the mat".split()
X = TOY_EMBED[[TOY_VOCAB.index(w) for w in SENT]]
COL = {"the": AMBER, "cat": BLUE, "sat": PLUM, "on": TEAL, "mat": VERM}

def heat(ax, T, vmax, ttl, labels=SENT, fmt="{:g}", fs=10, cm="RdBu_r", color=NAVY, hi=None, axes=True):
    ax.grid(False); ax.imshow(T, cmap=cm, vmin=-vmax, vmax=vmax)
    n = len(labels)
    ax.set_xticks(range(n), labels, fontsize=9); ax.set_yticks(range(n), labels, fontsize=9)
    ax.xaxis.tick_top()
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            v = float(T[i, j]) + 0.0
            ax.text(j, i, fmt.format(round(v, 3) + 0.0), ha="center", va="center", fontsize=fs,
                    color="white" if abs(v) > 0.6 * vmax else SLATE, weight="bold")
    if hi:
        for (i, j) in hi:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=NAVY, lw=3))
    ax.set_title(ttl, fontsize=11.5, color=color, pad=10)
    if axes:
        ax.set_ylabel("asking (row)"); ax.set_xlabel("answering (column)"); ax.xaxis.set_label_position("top")

# 7.1 the plain table
def f1():
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    heat(ax, X @ X.T, 2.6, "Thirty-six agreements of the plain word rows", hi=[(2, 1)])
    save(fig, P("fig07-01-plain-table.png"))

# 7.2 one cell: a row meets a column
def f2():
    fig, ax = canvas(12, 5.6)
    label(ax, 1, 53, "One entry of the table = one row of X meets one column of X turned on its side",
          NAVY, 12.5, ha="left", weight="bold")
    # X (6 x 2)
    for i, w in enumerate(SENT):
        y = 42 - i * 6
        c = PLUM if w == "sat" else GREY
        box(ax, 4, y, 7, 5, f"{X[i,0]:g}", c, fill=(w == "sat"), fs=11)
        box(ax, 12, y, 7, 5, f"{X[i,1]:g}", c, fill=(w == "sat"), fs=11)
        label(ax, 2.5, y + 2.5, w, COL[w], 10, ha="right", weight="bold")
    label(ax, 11.5, 50, "X  (6 rows, 2 columns)", SLATE, 10)
    # X.T (2 x 6)
    for j, w in enumerate(SENT):
        x = 30 + j * 8
        c = BLUE if j == 1 else GREY
        box(ax, x, 40, 7, 5, f"{X[j,0]:g}", c, fill=(j == 1), fs=11)
        box(ax, x, 34, 7, 5, f"{X[j,1]:g}", c, fill=(j == 1), fs=11)
        label(ax, x + 3.5, 47, w, COL[w], 10, weight="bold")
    label(ax, 53.5, 51, "X.T  (2 rows, 6 columns)", SLATE, 10)
    label(ax, 25, 37, "@", NAVY, 20, weight="bold")
    # the sum
    label(ax, 30, 24, "row sat (1, 1) meets column cat (0, 1):", PLUM, 12, ha="left", weight="bold")
    label(ax, 30, 17, "1 × 0  +  1 × 1  =  1", NAVY, 15, ha="left", weight="bold")
    label(ax, 30, 10, "it lands in row 2, column 1 of the 6 × 6 answer", SLATE, 10.5, ha="left")
    # small answer grid
    for i in range(6):
        for j in range(6):
            on = (i == 2 and j == 1)
            ax.add_patch(Rectangle((92 + j * 4.2, 36 - i * 4.2), 4, 4, fc=TINT[PLUM] if on else LIGHT,
                                   ec=PLUM if on else "#c9d1dc", lw=2 if on else .8))
    label(ax, 104.5, 44, "the answer: 6 × 6", SLATE, 10)
    label(ax, 98.2, 29.6, "1", PLUM, 10, weight="bold")
    arrow(ax, 60, 17, 90, 26, PLUM, lw=2, rad=-0.2)
    save(fig, P("fig07-02-one-cell.png"))

# 7.3 a matrix as a stack of lists; rows, columns, turned on its side
def f3():
    fig, ax = canvas(12, 5.4)
    label(ax, 1, 51, "A matrix is a stack of lists.  Its shape is (rows, columns).  .T turns it on its side",
          NAVY, 12.5, ha="left", weight="bold")
    for i, w in enumerate(SENT):
        y = 38 - i * 6
        box(ax, 10, y, 7, 5, f"{X[i,0]:g}", COL[w], fs=11)
        box(ax, 18, y, 7, 5, f"{X[i,1]:g}", COL[w], fs=11)
        label(ax, 8.5, y + 2.5, f"row {i}: {w}", COL[w], 9.5, ha="right", weight="bold")
    label(ax, 17.5, 46, "X, shape (6, 2)", SLATE, 10.5, weight="bold")
    ax.add_patch(Rectangle((9.3, 7.3), 8.4, 36.4, fill=False, ec=NAVY, lw=2, ls="--"))
    label(ax, 13.5, 4.5, "column 0", NAVY, 9.5); label(ax, 21.5, 4.5, "column 1", GREY, 9.5)
    arrow(ax, 30, 25, 44, 25, SLATE, lw=2)
    label(ax, 37, 29, ".T", NAVY, 14, weight="bold")
    for j, w in enumerate(SENT):
        x = 50 + j * 11
        box(ax, x, 26, 10, 5, f"{X[j,0]:g}", COL[w], fs=11)
        box(ax, x, 19, 10, 5, f"{X[j,1]:g}", COL[w], fs=11)
        label(ax, x + 5, 34, w, COL[w], 9.5, weight="bold")
    ax.add_patch(Rectangle((49.6, 25.6), 66.4, 5.8, fill=False, ec=NAVY, lw=2, ls="--"))
    label(ax, 83, 15, "X.T, shape (2, 6): row 0 of X.T is column 0 of X", SLATE, 10.5, weight="bold")
    label(ax, 83, 9, "entry X[5, 0] = −1 (mat's first number) is entry X.T[0, 5]", GREY, 10)
    save(fig, P("fig07-03-stack.png"))

# 7.4 shapes as dominoes
def f4():
    fig, ax = canvas(12, 5.6)
    label(ax, 1, 53, "The shape rule: the inside numbers must match; the answer takes the outside numbers",
          NAVY, 12.5, ha="left", weight="bold")
    rows = [((6, 2), (2, 6), "the toy table", True), ((7, 128), (128, 7), "Tiny's sentence", True),
            ((1024, 128), (128, 2048), "a window scored against every token", True), ((6, 2), (6, 2), "X @ X", False)]
    for r, (a, b, name, ok) in enumerate(rows):
        y = 41 - r * 11
        c = BLUE if ok else CORAL
        box(ax, 3, y, 15, 7, f"({a[0]}, {a[1]})", AMBER, fs=12)
        label(ax, 20.5, y + 3.5, "@", NAVY, 15, weight="bold")
        box(ax, 23, y, 15, 7, f"({b[0]}, {b[1]})", AMBER, fs=12)
        if ok:
            label(ax, 43, y + 3.5, "→", NAVY, 16, weight="bold")
            box(ax, 48, y, 17, 7, f"({a[0]}, {b[1]})", PLUM, fs=12)
            label(ax, 69, y + 3.5, f"{name}: inner {a[1]} = {b[0]}", SLATE, 10.5, ha="left")
        else:
            label(ax, 43, y + 3.5, "✗", CORAL, 16, weight="bold")
            box(ax, 48, y, 17, 7, "error", CORAL, fs=12)
            label(ax, 69, y + 3.5, f"{name}: inner {a[1]} ≠ {b[0]}", CORAL, 10.5, ha="left", weight="bold")
    save(fig, P("fig07-04-shape-rule.png"))

# 7.5 rotary table = X_rot @ X_rot.T
def f5():
    Xr = apply_rotary(X, range(6), speeds=TOY_TURN)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    heat(axs[0], X @ X.T, 2.6, "plain rows: X @ X.T")
    heat(axs[1], Xr @ Xr.T, 2.6, "rows turned 30° per seat: X_rot @ X_rot.T", fmt="{:.2f}", fs=8.5,
         labels=[f"{w}@{i}" for i, w in enumerate(SENT)])
    fig.suptitle("One line computes both tables; the right one is Chapter 6's lab table exactly",
                 x=0.02, ha="left", color=NAVY, weight="bold", fontsize=13)
    fig.tight_layout(); save(fig, P("fig07-05-rotary-table.png"))

# 7.6 star versus @
def f6():
    fig, ax = canvas(12, 5)
    label(ax, 1, 47, "Two different multiplications: * pairs up entries; @ adds up the pairs into one number",
          NAVY, 12.5, ha="left", weight="bold")
    for k, (v, c, x) in enumerate([((1, 1), PLUM, 4), ((0, 1), BLUE, 24)]):
        box(ax, x, 26, 7, 6, f"{v[0]}", c, fs=12); box(ax, x + 8, 26, 7, 6, f"{v[1]}", c, fs=12)
    label(ax, 9, 36, "sat", PLUM, 11, weight="bold"); label(ax, 29, 36, "cat", BLUE, 11, weight="bold")
    label(ax, 21, 29, "*", NAVY, 18, weight="bold")
    label(ax, 43, 29, "=", NAVY, 18, weight="bold")
    box(ax, 48, 26, 7, 6, "0", TEAL, fs=12); box(ax, 56, 26, 7, 6, "1", TEAL, fs=12)
    label(ax, 55.5, 36, "sat * cat: a list, shape (2,)", TEAL, 10.5, weight="bold")
    box(ax, 48, 8, 15, 6, "1", NAVY, fs=13)
    label(ax, 55.5, 18, "sat @ cat: one number, shape ()", NAVY, 10.5, weight="bold")
    arrow(ax, 64, 29, 75, 29, GREY); label(ax, 77, 29, "(sat * cat).sum()  =  1", SLATE, 11, ha="left", weight="bold")
    arrow(ax, 64, 11, 75, 11, GREY); label(ax, 77, 11, "the dot product of Chapter 4", SLATE, 11, ha="left", weight="bold")
    label(ax, 77, 20, "the same total, two routes", GREY, 10, ha="left")
    save(fig, P("fig07-06-star-vs-at.png"))

# 7.7 broadcasting
def f7():
    fig, ax = canvas(12, 5.6)
    label(ax, 1, 53, "Broadcasting: a stack plus a stack of the same shape, or plus one row copied to every row",
          NAVY, 12.5, ha="left", weight="bold")
    for i, w in enumerate(SENT):
        y = 40 - i * 6
        box(ax, 3, y, 13, 5, f"({X[i,0]:g}, {X[i,1]:g})", COL[w], fs=10)
        box(ax, 22, y, 13, 5, f"({TOY_POS[i,0]:g}, {TOY_POS[i,1]:g})", NAVY, fs=10)
        s = X[i] + TOY_POS[i]
        box(ax, 41, y, 13, 5, f"({s[0]:g}, {s[1]:g})", PLUM, fs=10)
        box(ax, 66, y, 13, 5, f"({X[i,0]:g}, {X[i,1]:g})", COL[w], fs=10)
        box(ax, 85, y, 13, 5, "(0.5, 0.5)", NAVY, fs=10, fill=(i == 0), ls="-" if i == 0 else "--")
        s2 = X[i] + 0.5
        box(ax, 104, y, 13, 5, f"({s2[0]:g}, {s2[1]:g})", PLUM, fs=10)
    for x0 in (16, 79): label(ax, x0 + 3, 32, "+", NAVY, 16, weight="bold")
    for x0 in (35, 98): label(ax, x0 + 3, 32, "=", NAVY, 16, weight="bold")
    label(ax, 28, 48, "X (6, 2) + TOY_POS (6, 2)", SLATE, 10.5, weight="bold")
    label(ax, 91, 48, "X (6, 2) + one row (2,)", SLATE, 10.5, weight="bold")
    label(ax, 91, 3, "the dashed rows are copies NumPy makes for you", GREY, 9.5)
    label(ax, 28, 3, "Chapter 5's add_positions, in one symbol", GREY, 9.5)
    save(fig, P("fig07-07-broadcast.png"))

# 7.8 timing bars
def f8():
    t = json.load(open(os.path.join(REC, "timing.json")))
    names = ["Chapter 6's loops", "one row at a time", "one line, W @ W.T", "one line, 32-bit"]
    vals = [t["loops"], t["rows"], t["line"], t["line32"]]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars = ax.barh(names[::-1], [v * 1000 for v in vals[::-1]], color=[TEAL, TEAL, GREY, VERM])
    ax.set_xscale("log"); ax.set_xlabel("milliseconds for the whole 1,024 × 1,024 table (log scale)")
    for b, v in zip(bars, vals[::-1]):
        ax.text(b.get_width() * 1.15, b.get_y() + b.get_height() / 2, f"{v * 1000:,.1f} ms", va="center", color=SLATE)
    ax.set_xlim(1, 1e5)
    ax.set_title(f"The same 268 million operations: one line is {t['loops'] / t['line']:,.0f} times faster than loops")
    ax.grid(axis="y", visible=False)
    save(fig, P("fig07-08-timing.png"))

# 7.9 speed by size
def f9():
    r = json.load(open(os.path.join(REC, "speed_by_size.json")))
    n = sorted(int(k) for k in r["all"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(n, [r["all"][str(k)] for k in n], "o-", color=BLUE, label=f"all {r['threads']} threads")
    ax.plot(n, [r["one"][str(k)] for k in n], "s-", color=TEAL, label="one thread")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(n, [str(k) for k in n])
    ax.set_xlabel("n: multiplying (n, n) by (n, n)"); ax.set_ylabel("billions of operations per second")
    ax2 = ax.twinx(); ax2.plot(n, [2 * k ** 3 / (3 * k * k) for k in n], ":", color=AMBER, label="operations per number moved")
    ax2.set_yscale("log"); ax2.set_ylabel("operations per number moved", color=AMBER); ax2.grid(False)
    ax.legend(loc="upper left", fontsize=9.5); ax2.legend(loc="lower right", fontsize=9.5)
    ax.set_title("Bigger multiplications run faster per operation: more work for every number fetched")
    save(fig, P("fig07-09-speed-by-size.png"))

# 7.10 roofline
def f10():
    tops, gbs = 989, 3350
    ai = np.logspace(-0.3, 3.6, 200)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ai, np.minimum(tops, ai * gbs / 1000), color=NAVY, lw=3)
    ridge = tops * 1000 / gbs
    ax.axvline(ridge, color=GREY, ls="--"); ax.text(ridge * 1.08, 4, f"break-even\n{ridge:.0f} operations per byte", color=GREY)
    for rows, c, dx, dy, ha in [(1, CORAL, 1.12, .62, "left"), (64, AMBER, 1.12, .62, "left"),
                                (512, PLUM, .9, .55, "right"), (4096, BLUE, 1.0, .55, "left")]:
        x = intensity(rows, 4096, 4096); y = min(tops, x * gbs / 1000)
        ax.plot([x], [y], "o", color=c, ms=10)
        ax.text(x * dx, y * dy, f"{rows} row{'s' if rows > 1 else ''} at a time", color=c, weight="bold", ha=ha)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("operations per byte fetched from memory"); ax.set_ylabel("best possible trillion operations per second")
    ax.text(1.2, 500, "limited by memory:\nthe slope", color=SLATEBLUE, weight="bold")
    ax.text(900, 1700, "limited by the multiply units: the flat roof", color=NAVY, weight="bold", ha="center")
    ax.set_ylim(1, 3000)
    ax.set_title("The roofline for one data-centre chip (published H100 SXM figures, 16-bit): rows @ (4096, 4096)")
    save(fig, P("fig07-10-roofline.png"))

# 7.11 tiles
def f11():
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axs[0]; ax.set_xlim(0, 1024); ax.set_ylim(1024, 0); ax.set_aspect("equal"); ax.grid(False)
    for k in range(0, 1025, 128):
        ax.axhline(k, color="#c9d1dc", lw=.8); ax.axvline(k, color="#c9d1dc", lw=.8)
    ax.add_patch(Rectangle((256, 384), 128, 128, fc=TINT[PLUM], ec=PLUM, lw=2.5))
    ax.add_patch(Rectangle((0, 384), 1024, 128, fill=False, ec=AMBER, lw=2, ls="--"))
    ax.add_patch(Rectangle((256, 0), 128, 1024, fill=False, ec=BLUE, lw=2, ls="--"))
    ax.set_title("the 1,024 × 1,024 table in 64 tiles of 128 × 128", fontsize=11)
    ax.set_xlabel("one tile needs 128 asking rows (amber)\nand 128 answering rows (blue)", fontsize=9)
    ax.set_xticks([0, 512, 1024]); ax.set_yticks([0, 512, 1024])
    ax = axs[1]
    sizes = [1, 8, 32, 128, 512, 1024]
    reads = [(1024 // s) ** 2 * 2 * s * 128 for s in sizes]
    ax.bar([str(s) for s in sizes], reads, color=[CORAL, AMBER, AMBER, TEAL, TEAL, TEAL])
    ax.set_yscale("log"); ax.set_xlabel("tile side"); ax.set_ylabel("numbers read from far memory")
    for i, r in enumerate(reads): ax.text(i, r * 1.2, f"{r:,}", ha="center", fontsize=8.5, color=SLATE)
    ax.set_title("bigger tiles, fewer trips: 1,024 times the inputs down to once", fontsize=11)
    fig.suptitle("Tiling: the same answer, one block at a time, with each fetched number used many times",
                 x=0.02, ha="left", color=NAVY, weight="bold", fontsize=13)
    fig.tight_layout(); save(fig, P("fig07-11-tiles.png"))

# 7.12 a batch is a stack of stacks
def f12():
    fig, ax = canvas(12, 5.6)
    label(ax, 1, 53, "A batch: three sentences, six rows of two numbers each, shape (3, 6, 2)", NAVY, 12.5, ha="left", weight="bold")
    texts = ["the cat sat on the mat", "the mat sat on the cat", "the rabbit looked at · ·"]
    for s_, t in enumerate(texts):
        x0, c = 2 + s_ * 26, [BLUE, TEAL, AMBER][s_]
        ax.add_patch(Rectangle((x0, 4), 24, 42, fc="white", ec=c, lw=2))
        label(ax, x0 + 12, 43, f"sentence {s_}  (batch[{s_}])", c, 9.5, weight="bold")
        for i, w in enumerate(t.split()):
            pad = w == "·"; y = 36 - i * 5.6
            ax.text(x0 + 2, y, "<pad>" if pad else w, fontsize=10, color=GREY if pad else COL.get(w, SLATE), weight="bold")
            v = (0, 0) if pad else TOY_EMBED[TOY_VOCAB.index(w)]
            ax.text(x0 + 14, y, f"({v[0]:g}, {v[1]:g})", fontsize=10, color=SLATE)
    label(ax, 99, 40, "batch @ batch.transpose(0, 2, 1)", NAVY, 11.5, weight="bold")
    label(ax, 99, 34, "(3, 6, 2) @ (3, 2, 6)  →  (3, 6, 6)", PLUM, 12, weight="bold")
    label(ax, 99, 26, "three tables in one call;", SLATE, 10.5)
    label(ax, 99, 21.5, "the first number, 3, rides along", SLATE, 10.5)
    label(ax, 99, 12, "short sentences are padded with (0, 0):", GREY, 10)
    label(ax, 99, 7.5, "they agree 0 with everything", GREY, 10)
    save(fig, P("fig07-12-batch.png"))

# 7.13 batching: time per row
def f13():
    t = json.load(open(os.path.join(REC, "batches.json")))
    b = [int(k) for k in t if k != "measured"]; b.sort()
    y = np.array([t[str(k)] for k in b])
    beta, alpha = np.polyfit(np.array(b, float), y, 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axs[0]
    ax.plot(b, [t[str(k)] / k * 1e6 for k in b], "o-", color=PLUM, label="measured")
    bb = np.logspace(0, 3.02, 100)
    ax.plot(bb, (alpha / bb + beta) * 1e6, ":", color=GREY, label=f"fixed {alpha*1e6:.0f} µs ÷ rows + {beta*1e6:.1f} µs")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(b, [str(k) for k in b], fontsize=8.5)
    ax.set_xlabel("rows scored in one call"); ax.set_ylabel("microseconds per row"); ax.legend(fontsize=9)
    ax.set_title("cost per row falls, then flattens", fontsize=11)
    ax = axs[1]
    ax.plot(b, [k / t[str(k)] for k in b], "o-", color=TEAL)
    ax.set_xscale("log", base=2); ax.set_xticks(b, [str(k) for k in b], fontsize=8.5)
    ax.set_xlabel("rows scored in one call"); ax.set_ylabel("rows per second")
    ax.set_title("throughput climbs to a plateau", fontsize=11)
    ax2 = ax.twinx(); ax2.plot(b, [t[str(k)] * 1000 for k in b], "s--", color=VERM, ms=4); ax2.grid(False)
    ax2.set_ylabel("milliseconds per call (each row waits this long)", color=VERM)
    fig.suptitle("Batching on one CPU core: (rows, 128) @ (128, 2048), Tiny's tied scores", x=0.02, ha="left",
                 color=NAVY, weight="bold", fontsize=13)
    fig.tight_layout(); save(fig, P("fig07-13-batching.png"))

# 7.14 the silent broadcast
def f14():
    lengths = (X * X).sum(axis=1)
    D = lengths - lengths.reshape(6, 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axs[0]; ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(0, 9.4, "lengths, shape (6,)", color=AMBER, weight="bold")
    for j, v in enumerate(lengths): ax.text(0.3 + j * 1.2, 8.4, f"{v:g}", color=AMBER, fontsize=12, weight="bold")
    ax.text(0, 6.8, "column, shape (6, 1)", color=BLUE, weight="bold")
    for i, v in enumerate(lengths): ax.text(0.6, 5.9 - i * .9, f"{v:g}", color=BLUE, fontsize=12, weight="bold")
    ax.text(2.2, 3.4, "lengths − column\nruns without a word\nand returns (6, 6)", color=CORAL, fontsize=12, weight="bold")
    ax = axs[1]
    heat(ax, D, 1.5, "BROKEN: 36 numbers where six zeros were wanted", color=CORAL, axes=False)
    fig.tight_layout(); save(fig, P("fig07-14-silent-broadcast.png"))

for f in (f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14):
    f()
print("chapter 7 figures written to", OUT)
