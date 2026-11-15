"""Figures for Chapter 5 (Position), computed with the same code as the listings.
Run from the project root: python3 03-figures/src/ch05_figs.py"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code")); sys.path.insert(0, os.path.join(ROOT, "04-code", "ch04"))
from style import *
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from tiny.tokenizer import TOY_VOCAB, load_bpe, TINY_VOCAB_SIZE
from tiny.embed import (TOY_EMBED, TOY_POS, clock, clock_speeds, sinusoidal, add_positions, dot, length, cosine,
                        Embedding, PositionTable, embed_with_positions)
import _gpt2
setup()
OUT = os.path.join(ROOT, "03-figures", "ch05"); os.makedirs(OUT, exist_ok=True)
P = lambda n: os.path.join(OUT, n)
SENT = "the cat sat on the mat".split()
SCR = "the mat sat on the cat".split()
ROW = {w: TOY_EMBED[i] for i, w in enumerate(TOY_VOCAB)}
COL = {"the": AMBER, "cat": BLUE, "sat": PLUM, "on": TEAL, "mat": VERM}
X = np.array([ROW[w] for w in SENT])
LP = add_positions(X, TOY_POS)
CLK = np.array([clock(k) for k in range(6)])
CP = add_positions(X, CLK)
WPE = _gpt2.tensor("wpe.weight").astype(float)
WTE_MED = float(np.median(np.linalg.norm(_gpt2.tensor("wte.weight"), axis=1)))

def plane(ax, lim=2.4):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, color="#9aa5b1", lw=1); ax.axvline(0, color="#9aa5b1", lw=1)
    ax.set_xlabel("first number"); ax.set_ylabel("second number")

def vec(ax, v, color, text=None, start=(0, 0), lw=2.6, off=(0.08, 0.08), fs=10.5, ls="-", alpha=1.0):
    ax.annotate("", xy=(start[0] + v[0], start[1] + v[1]), xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=15, ls=ls, alpha=alpha))
    if text:
        ax.text(start[0] + v[0] + off[0], start[1] + v[1] + off[1], text, color=color, fontsize=fs, weight="bold", alpha=alpha)

# 5.1 the bag
def f1():
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, words in zip(axs, (SENT, SCR)):
        plane(ax, 1.7)
        for w in ["the", "cat", "sat", "on", "mat"]:
            vec(ax, ROW[w], COL[w], w + (" x2" if w == "the" else ""), off=(0.06, -0.17) if w == "on" else (0.06, 0.06))
        ax.set_title("\u201c" + " ".join(words) + "\u201d")
    fig.suptitle("Two sentences, one bag: without positions both give exactly the same six arrows",
                 x=0.02, ha="left", color=NAVY, weight="bold", fontsize=13)
    fig.tight_layout(); save(fig, P("fig05-01-bag.png"))

# 5.2 tip to tail
def f2():
    fig, ax = plt.subplots(figsize=(6.6, 5.6)); plane(ax, 1.9)
    ax.set_xlim(-0.6, 2.3); ax.set_ylim(-0.4, 1.0)
    vec(ax, ROW["the"], AMBER, "the (1, 0)", off=(-0.35, -0.16))
    vec(ax, TOY_POS[0], NAVY, "position 0 (0.5, 0.5)", start=ROW["the"], off=(-0.3, 0.1))
    vec(ax, LP[0], PLUM, "the@0 = (1.5, 0.5)", off=(0.06, -0.04), lw=3.2)
    vec(ax, TOY_POS[4], NAVY, None, start=ROW["the"], ls="--")
    ax.text(0.3, 0.76, "position 4 (-0.5, 0.5)", color=NAVY, fontsize=10.5, weight="bold")
    vec(ax, LP[4], CORAL, None, lw=3.2)
    ax.text(-0.55, 0.62, "the@4 = (0.5, 0.5)", color=CORAL, fontsize=10.5, weight="bold")
    ax.set_title("Same word, two places: add the position arrow tip to tail")
    save(fig, P("fig05-02-tip-to-tail.png"))

# 5.3 two tables
def f3():
    fig, ax = canvas(12.5, 6.4)
    label(ax, 1, 61, "Two lookups, one addition: which token (amber id) and where it stands (position), added row by row", NAVY, 12, ha="left", weight="bold")
    ids = [TOY_VOCAB.index(w) for w in SENT]
    label(ax, 9, 54, "token rows (learned)", NAVY, 10, weight="bold")
    label(ax, 37, 54, "position rows (learned)", NAVY, 10, weight="bold")
    label(ax, 70, 54, "sum: what the model reads", PLUM, 10, weight="bold")
    for k, (w, i) in enumerate(zip(SENT, ids)):
        y = 45 - k * 7.4
        box(ax, 1, y, 16, 6, f"({ROW[w][0]:g}, {ROW[w][1]:g})", NAVY, fs=11.5, sub=f"{w}  (id {i})")
        label(ax, 21, y + 3, "+", SLATE, 18, weight="bold")
        box(ax, 26, y, 22, 6, f"({TOY_POS[k][0]:g}, {TOY_POS[k][1]:g})", BLUE, fs=11.5, sub=f"position {k}")
        label(ax, 53, y + 3, "=", SLATE, 18, weight="bold")
        box(ax, 59, y, 22, 6, f"({LP[k][0]:g}, {LP[k][1]:g})", PLUM, fs=11.5, sub=f"{w}@{k}")
    box(ax, 88, 26, 35, 20, "Tiny: 1,024 rows", NAVY, fs=11, sub="x 128 numbers = 131,072")
    label(ax, 105.5, 22, "GPT-2: 1,024 x 768", SLATE, 10)
    label(ax, 105.5, 18.5, "= 786,432 numbers", SLATE, 10)
    label(ax, 62, 1.5, "the two 'the' rows were identical; now (1.5, 0.5) and (0.5, 0.5)", CORAL, 10.5, weight="bold")
    save(fig, P("fig05-03-two-tables.png"))

# 5.4 nudged arrows
def f4():
    fig, ax = plt.subplots(figsize=(6.8, 6.4)); plane(ax, 1.9)
    offs = [(0.05, 0.05), (0.05, 0.05), (-0.2, 0.14), (0.06, -0.14), (-0.62, 0.08), (0.05, 0.05)]
    for k, w in enumerate(SENT):
        vec(ax, X[k], COL[w], None, lw=1.6, alpha=0.3)
        ax.plot([X[k][0], LP[k][0]], [X[k][1], LP[k][1]], ls=":", color=NAVY, lw=1.6)
        vec(ax, LP[k], COL[w], f"{w}@{k}", off=offs[k])
    ax.set_title("Six arrows, nudged by position (faint = word only)")
    save(fig, P("fig05-04-nudged.png"))

# 5.5 clock face
def f5():
    fig, ax = plt.subplots(figsize=(9, 6.0)); ax.set_aspect("equal")
    t = np.linspace(0, 2 * np.pi, 300); ax.plot(np.cos(t), np.sin(t), color="#9aa5b1", lw=1.5)
    for h in range(12):
        x, y = clock(h); ax.text(1.12 * x, 1.12 * y, str(h or 12), ha="center", va="center", color=GREY, fontsize=10)
    for k in range(6):
        x, y = clock(k)
        vec(ax, (x, y), SERIES[k % len(SERIES)], None, lw=2.2)
        ax.text(1.5, 0.9 - k * 0.3, f"position {k}: ({x:.2f}, {y:.2f})", color=SERIES[k % len(SERIES)], fontsize=10.5, weight="bold")
    ax.set_xlim(-1.3, 2.75); ax.set_ylim(-1.3, 1.3); ax.axhline(0, color="#e3e7ee"); ax.axvline(0, color="#e3e7ee")
    ax.set_xlabel("across (first number)"); ax.set_ylabel("up (second number)")
    ax.set_title("One clock, one hour per token: position k = hand at k o'clock")
    save(fig, P("fig05-05-clock-face.png"))

# 5.6 waves
def f6():
    k = np.arange(0, 25); t = np.linspace(0, 24, 400)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, np.sin(np.radians(30 * t)), color=BLUE, label="across = sine of the angle")
    ax.plot(t, np.cos(np.radians(30 * t)), color=VERM, label="up = cosine of the angle")
    ax.scatter(k, np.sin(np.radians(30 * k)), color=BLUE, zorder=3); ax.scatter(k, np.cos(np.radians(30 * k)), color=VERM, zorder=3)
    ax.axvline(12, color=GREY, ls="--", lw=1); ax.text(12.2, 1.08, "position 12 = position 0", color=GREY, fontsize=10)
    ax.set_xlabel("position"); ax.set_ylim(-1.25, 1.3); ax.legend(loc="lower left", ncol=2)
    ax.set_title("Follow the tip of the hand as the tokens go by: two waves, a lap every 12 tokens")
    save(fig, P("fig05-06-waves.png"))

# 5.7 two clocks
def f7():
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, k in zip(axs, (0, 4, 12)):
        ax.set_aspect("equal"); t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(t), np.sin(t), color="#9aa5b1", lw=1.2); ax.axis("off")
        f, s = clock(k, 30), clock(k, 7.5)
        vec(ax, f, BLUE, "fast", lw=2.8, off=(0.05, 0.05)); vec(ax, [0.6 * v for v in s], PLUM, "slow", lw=4, off=(0.05, -0.15))
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.set_title(f"position {k}: ({f[0]:.2f}, {f[1]:.2f}, {s[0]:.2f}, {s[1]:.2f})", fontsize=11)
    fig.suptitle("Two hands on one dial: the fast hand (30° per token) repeats at 12, the slow one (7.5°) has moved on",
                 x=0.02, ha="left", color=NAVY, weight="bold", fontsize=12.5)
    fig.tight_layout(); save(fig, P("fig05-07-two-clocks.png"))

# 5.8 heatmap of Tiny's sinusoidal table
def f8():
    S = sinusoidal(128)
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(S, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("place in the row (pairs = one clock each: fast on the left, slow on the right)")
    ax.set_ylabel("position"); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.025)
    ax.set_title("Tiny's sinusoidal rows, positions 0-127: 64 clocks, each slower than the last")
    save(fig, P("fig05-08-sinusoid-heatmap.png"))

# 5.9 agreement depends only on gap
def f9():
    S = sinusoidal(256); D = S @ S.T
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
    im = axs[0].imshow(D, cmap="RdBu_r", vmin=-64, vmax=64); axs[0].grid(False)
    axs[0].set_xlabel("position"); axs[0].set_ylabel("position"); axs[0].set_title("dot product of every pair")
    fig.colorbar(im, ax=axs[0], fraction=0.045)
    gaps = np.arange(0, 256)
    for a, c in ((0, BLUE), (100, VERM), (600, TEAL)):
        T = sinusoidal(a + 256)
        axs[1].plot(gaps, [dot(T[a], T[a + g]) for g in gaps], color=c, lw=2.2 if a == 0 else 1.4,
                    ls="-" if a == 0 else "--", label=f"starting at position {a}")
    axs[1].set_xlabel("gap between the two positions"); axs[1].set_ylabel("dot product"); axs[1].legend()
    axs[1].set_title("the three curves lie on top of each other")
    fig.suptitle("Sinusoidal rows: agreement depends only on how far apart two positions are",
                 x=0.02, ha="left", color=NAVY, weight="bold", fontsize=13)
    fig.tight_layout(); save(fig, P("fig05-09-agreement-by-gap.png"))

# 5.10 compare toy
OFF10 = {"learn": [(0.05, 0.05), (0.05, 0.05), (-0.25, 0.14), (0.06, -0.14), (-0.75, 0.0), (0.05, 0.05)],
         "clock": [(-0.55, 0.08), (0.05, 0.05), (0.05, 0.05), (0.06, -0.14), (0.05, -0.2), (-0.6, 0.12)]}
def f10():
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.8))
    for ax, Y, ttl in ((axs[0], LP, "learned (hand-chosen) rows"), (axs[1], CP, "clock rows, 30° per token")):
        plane(ax, 2.3)
        for k, w in enumerate(SENT):
            vec(ax, X[k], COL[w], None, lw=1.4, alpha=0.3)
            vec(ax, Y[k], COL[w], f"{w}@{k}", off=OFF10[ttl[:5]][k])
        ax.set_title(ttl)
    fig.suptitle("The running sentence with each kind of position (faint = word only)", x=0.02, ha="left",
                 color=NAVY, weight="bold", fontsize=13)
    fig.tight_layout(); save(fig, P("fig05-10-compare-toy.png"))

# 5.11 GPT-2 lengths
def f11():
    L = np.linalg.norm(WPE, axis=1)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(L, color=NAVY, lw=1.6, label="length of GPT-2's position row")
    ax.axhline(WTE_MED, color=AMBER, ls="--", lw=1.4, label=f"typical token row ({WTE_MED:.2f})")
    ax.annotate(f"position 0: {L[0]:.2f}", (0, L[0]), (60, 9.2), color=VERM, weight="bold", arrowprops=dict(arrowstyle="->", color=VERM))
    ax.annotate(f"position 1023: {L[1023]:.3f}", (1023, L[1023]), (790, 1.2), color=CORAL, weight="bold", arrowprops=dict(arrowstyle="->", color=CORAL))
    ax.set_xlabel("position"); ax.set_ylabel("length"); ax.legend(loc="upper right"); ax.set_ylim(0, 10.6)
    ax.set_title("GPT-2's learned position rows are as long as its word rows, except at the two ends")
    save(fig, P("fig05-11-gpt2-lengths.png"))

# 5.12 cosine maps
def f12():
    U = WPE / np.linalg.norm(WPE, axis=1, keepdims=True); C = U @ U.T
    S = sinusoidal(1024) / 8.0; D = S @ S.T
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, M, ttl in ((axs[0], C, "GPT-2: learned (1,024 x 768)"), (axs[1], D, "Tiny: sinusoidal (1,024 x 128)")):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False); ax.set_title(ttl)
        ax.set_xlabel("position"); ax.set_ylabel("position")
    fig.colorbar(im, ax=axs, fraction=0.025)
    fig.suptitle("Cosine between every pair of position rows: both put near positions close together",
                 x=0.02, ha="left", color=NAVY, weight="bold", fontsize=13)
    save(fig, P("fig05-12-cosine-maps.png"))

# 5.13 GPT-2 columns
def f13():
    cols = np.argsort(-WPE[1:1023].std(axis=0))[:4]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for c, colour in zip(cols, (BLUE, VERM, TEAL, PLUM)):
        ax.plot(np.arange(1, 1023), WPE[1:1023, c], color=colour, lw=1.4, label=f"place {c}")
    ax.set_xlabel("position"); ax.set_ylabel("value"); ax.legend(ncol=4, loc="lower left")
    ax.set_title("Four places in GPT-2's position rows, read down the table: slow waves nobody wrote down")
    save(fig, P("fig05-13-gpt2-waves.png"))

# 5.14 past the edge
def f14():
    S = sinusoidal(4096) / 8.0; T = S[:1024]
    near = []
    for p in range(4096):
        c = T @ S[p]
        if p < 1024: c[p] = -2
        near.append(c.max())
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.axvspan(0, 1023, color=TINT[NAVY]); ax.text(30, 1.02, "trained positions 0-1,023", color=NAVY, weight="bold")
    ax.plot(near, color=BLUE, lw=1.3, label="sinusoidal: cosine with the closest trained row")
    ax.plot([0, 1023], [-0.15, -0.15], color=NAVY, lw=6, solid_capstyle="butt", label="learned table: has a row")
    ax.plot([1024, 4095], [-0.15, -0.15], color=CORAL, lw=6, solid_capstyle="butt", label="learned table: IndexError")
    ax.set_ylim(-0.3, 1.1); ax.set_xlabel("position"); ax.legend(loc="center right")
    ax.set_title("Past the edge: a learned table stops; sinusoidal rows go on, but look like nothing seen in training")
    save(fig, P("fig05-14-past-edge.png"))

# 5.15 padding
def f15():
    enc = tiktoken.get_encoding("gpt2"); E = _gpt2.tensor("wte.weight").astype(float)
    ids = enc.encode("The rabbit looked at"); pad = 3
    cs = [cosine(E[t] + WPE[k], E[t] + WPE[k + pad]) for k, t in enumerate(ids)]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar([enc.decode([t]) for t in ids], cs, color=[CORAL, VERM, AMBER, AMBER])
    for b, c in zip(bars, cs): ax.text(b.get_x() + b.get_width() / 2, c + 0.02, f"{c:.3f}", ha="center", color=SLATE, weight="bold")
    ax.set_ylim(0, 1.1); ax.set_ylabel("cosine: right row vs shifted row")
    ax.set_title("Left padding of 3 without shifting the positions: the first token changes most")
    save(fig, P("fig05-15-padding.png"))

# 5.16 Signature 1 extended
def f16():
    fig, ax = canvas(12, 4.8)
    ids = [TOY_VOCAB.index(w) for w in SENT]
    label(ax, 2, 46, "Signature 1, extended: every token now carries its place as well as its identity", NAVY, 12, ha="left", weight="bold")
    for i, (t, d) in enumerate(zip(SENT, ids)):
        x = 4 + i * 19
        box(ax, x, 33, 15, 7, t, NAVY, fs=13)
        label(ax, x + 7.5, 42, f"position {i}", GREY, 9)
        box(ax, x, 23, 7, 6, str(d), AMBER, fs=11)
        box(ax, x + 8, 23, 7, 6, str(i), BLUE, fs=11)
        box(ax, x - 1, 8, 17, 8, f"({LP[i][0]:g}, {LP[i][1]:g})", PLUM, fs=11.5,
            sub=f"({X[i][0]:g},{X[i][1]:g})+({TOY_POS[i][0]:g},{TOY_POS[i][1]:g})")
        arrow(ax, x + 3.5, 23, x + 6, 16.3, AMBER, lw=1.2, ms=9); arrow(ax, x + 11.5, 23, x + 9, 16.3, BLUE, lw=1.2, ms=9)
    label(ax, 60, 3, "amber = token id, blue = position; six different rows for six tokens: the two 'the's no longer look alike", CORAL, 10.5, weight="bold")
    save(fig, P("fig05-16-signature-1-extended.png"))

# 5.17 Tiny block
def f17():
    tok = load_bpe(os.path.join(ROOT, "04-code", "tiny", "tiny_bpe_2048.txt"))
    emb = Embedding(TINY_VOCAB_SIZE, seed=1865); pos = PositionTable(seed=1865)
    ids = tok.encode(" ".join(SENT)); T = emb(ids); Q = pos(range(len(ids))); B = T + Q
    pieces = [tok.piece(i) if hasattr(tok, "piece") else str(i) for i in ids]
    pieces = [p.decode("utf-8", "replace") if isinstance(p, bytes) else str(p) for p in pieces]
    pieces = [p.replace(" ", "␣") for p in pieces]
    fig, axs = plt.subplots(3, 1, figsize=(12, 6.6), sharex=True)
    for ax, M, ttl in zip(axs, (T, Q, B), ("token rows (spread 0.02)", "position rows 0-6 (spread 0.01)", "sum: Tiny's input block, 7 x 128")):
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-0.06, vmax=0.06); ax.grid(False)
        ax.set_yticks(range(len(ids))); ax.set_yticklabels(pieces, fontsize=8.5); ax.set_title(ttl, fontsize=11.5)
    axs[-1].set_xlabel("place in the row (0-127)")
    fig.colorbar(im, ax=axs, fraction=0.02)
    save(fig, P("fig05-17-tiny-block.png"))

# 5.18 breakage
def f18():
    flat = np.tile([0.5, 0.5], (6, 1)); F = add_positions(X, flat)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.6))
    for ax, Y, ttl, col in ((axs[0], LP, "different position rows: six arrows", NAVY), (axs[1], F, "BREAKAGE: every position (0.5, 0.5)", CORAL)):
        plane(ax, 2.0)
        for k, w in enumerate(SENT):
            vec(ax, Y[k], COL[w] if col == NAVY else CORAL, f"{w}@{k}",
                off=(0.05, 0.05) if not (col == CORAL and k == 4) else (0.05, -0.22))
        ax.set_title(ttl, color=col)
    fig.suptitle("The same nudge for everyone is no information: the two 'the's land on one arrow again",
                 x=0.02, ha="left", color=NAVY, weight="bold", fontsize=12.5)
    fig.tight_layout(); save(fig, P("fig05-18-breakage-flat.png"))

for f in (f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18):
    f()
print("chapter 5 figures written to", OUT)
