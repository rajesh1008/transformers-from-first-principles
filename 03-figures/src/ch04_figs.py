"""Figures for Chapter 4 (Embeddings), computed with the same code as the listings.
Run from the project root: python3 03-figures/src/ch04_figs.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code")); sys.path.insert(0, os.path.join(ROOT, "04-code", "ch04"))
from style import *
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from tiny.tokenizer import TOY_VOCAB, load_bpe, TINY_VOCAB_SIZE
from tiny.embed import TOY_EMBED, Embedding, dot, length, cosine
import _gpt2
from nudge import train, nearest, squash
setup()
OUT = os.path.join(ROOT, "03-figures", "ch04"); os.makedirs(OUT, exist_ok=True)
P = lambda n: os.path.join(OUT, n)
SENT = "the cat sat on the mat".split()
ROW = {w: TOY_EMBED[i] for i, w in enumerate(TOY_VOCAB)}
COL = {"the": AMBER, "cat": BLUE, "sat": PLUM, "on": TEAL, "mat": VERM}
enc = tiktoken.get_encoding("gpt2")
W = _gpt2.tensor("wte.weight").astype(float)
NW = W / np.linalg.norm(W, axis=1, keepdims=True)

def plane(ax, lim=2.4, ticks=True):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, color="#9aa5b1", lw=1); ax.axvline(0, color="#9aa5b1", lw=1)
    ax.set_xlabel("first number"); ax.set_ylabel("second number")

def vec(ax, v, color, text=None, start=(0, 0), lw=2.6, off=(0.12, 0.12), fs=11, ls="-"):
    ax.annotate("", xy=(start[0] + v[0], start[1] + v[1]), xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=16, ls=ls))
    if text:
        ax.text(start[0] + v[0] + off[0], start[1] + v[1] + off[1], text, color=color, fontsize=fs, weight="bold")

# 4.1 lookup
def f1():
    fig, ax = canvas(12, 6.2)
    label(ax, 1, 59, "An embedding layer is a lookup: each id picks out one row of the table", NAVY, 12.5, ha="left", weight="bold")
    ids = [0, 1, 2, 3, 0, 4]
    label(ax, 7, 52, "token ids (data)", AMBER, 10, weight="bold")
    for k, (i, w) in enumerate(zip(ids, SENT)):
        box(ax, 2, 44 - k * 7, 10, 5.5, str(i), AMBER, fs=12, sub=w)
    label(ax, 47, 52, "the table: 12 rows x 2 numbers (learned)", NAVY, 10, weight="bold")
    for r, w in enumerate(TOY_VOCAB):
        y = 47.5 - r * 3.9
        hit = r in ids
        box(ax, 34, y, 26, 3.3, f"{r:>2}  {w:<7}  ({ROW[w][0]:g}, {ROW[w][1]:g})", NAVY if hit else GREY,
            fs=9.5, weight="bold" if hit else "normal", r=0.8, lw=1.5 if hit else 0.8)
    label(ax, 92, 52, "six rows out (one per token)", PLUM, 10, weight="bold")
    for k, (i, w) in enumerate(zip(ids, SENT)):
        y = 44 - k * 7
        box(ax, 82, y, 20, 5.5, f"({ROW[w][0]:g}, {ROW[w][1]:g})", PLUM, fs=12, sub=w)
        arrow(ax, 12.5, y + 2.75, 33.5, 47.5 - i * 3.9 + 1.65, AMBER, lw=1.2)
        arrow(ax, 60.5, 47.5 - i * 3.9 + 1.65, 81.5, y + 2.75, PLUM, lw=1.2)
    label(ax, 60, 1.5, "id 0 appears twice, so the same row comes out twice: a lookup ignores position", SLATE, 10)
    save(fig, P("fig04-01-lookup.png"))

# 4.2 the six arrows
def f2():
    fig, ax = plt.subplots(figsize=(6.2, 6.2)); plane(ax, 1.7)
    offs = {"the": (0.06, 0.08), "cat": (0.08, 0.05), "sat": (0.06, 0.06), "on": (0.06, -0.15), "mat": (-0.3, 0.08)}
    for w in ["the", "cat", "sat", "on", "mat"]:
        vec(ax, ROW[w], COL[w], f"{w} ({ROW[w][0]:g}, {ROW[w][1]:g})" + (" x2" if w == "the" else ""), off=offs[w])
    ax.set_title("The running sentence as arrows: six tokens, five arrows")
    save(fig, P("fig04-02-six-arrows.png"))

# 4.3 adding and stretching
def f3():
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    ax = axs[0]; plane(ax, 1.6)
    vec(ax, ROW["cat"], BLUE, "cat"); vec(ax, ROW["the"], AMBER, "the", start=ROW["cat"], off=(-0.1, 0.12))
    vec(ax, ROW["sat"], PLUM, "cat + the = (1, 1)", off=(-0.6, -0.35))
    ax.set_title("Add: walk one arrow, then the next")
    ax = axs[1]; plane(ax, 3.6)
    vec(ax, 3 * ROW["sat"], GREY, "3 x sat = (3, 3)", lw=5, off=(-2.2, 0.15)); vec(ax, ROW["sat"], PLUM, "sat", off=(0.1, -0.3))
    ax.set_title("Stretch: multiply every number")
    ax = axs[2]; plane(ax, 1.6)
    vec(ax, ROW["on"], TEAL, "on (1, -1)"); vec(ax, -ROW["on"], VERM, "-1 x on = (-1, 1)", off=(-0.2, 0.15))
    ax.set_title("Flip: multiply by -1")
    fig.tight_layout(); save(fig, P("fig04-03-add-stretch.png"))

# 4.4 length
def f4():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, v, name in [(axs[0], (1, 1), "sat"), (axs[1], (3, 4), "(3, 4)")]:
        lim = max(v) + 0.8; ax.set_xlim(-0.3, lim); ax.set_ylim(-0.3, lim); ax.set_aspect("equal")
        ax.axhline(0, color="#9aa5b1"); ax.axvline(0, color="#9aa5b1")
        ax.plot([0, v[0]], [0, 0], color=AMBER, lw=3); ax.plot([v[0], v[0]], [0, v[1]], color=AMBER, lw=3)
        vec(ax, v, PLUM, None)
        ax.text(v[0] / 2, -0.25, f"across {v[0]}", color=AMBER, ha="center", weight="bold")
        ax.text(v[0] + 0.1, v[1] / 2, f"up {v[1]}", color=AMBER, weight="bold")
        L = (v[0] ** 2 + v[1] ** 2) ** 0.5
        ax.text(0.05, lim - 0.15, f"length = sqrt({v[0]}² + {v[1]}²) = {L:.4g}", color=PLUM, weight="bold", ha="left", va="top")
        ax.set_title(f"Length of {name}: Pythagoras")
    fig.tight_layout(); save(fig, P("fig04-04-length.png"))

# 4.5 agreement cases
def f5():
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (a, b) in zip(axs, [("sat", "cat"), ("sat", "on"), ("on", "mat")]):
        plane(ax, 1.6); vec(ax, ROW[a], COL[a], a); vec(ax, ROW[b], COL[b], b, off=(0.08, -0.2) if b == "on" else (0.08, 0.08))
        d = dot(ROW[a], ROW[b]); c = cosine(ROW[a], ROW[b])
        ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
        kind = {1: "same general way", 0: "at right angles", -2: "opposite ways"}[int(d)]
        ax.set_title(f"{a} . {b} = {d:g}  ({kind})", fontsize=11.5)
        ax.text(-1.5, -1.45, f"cosine {c:.3f}, angle {ang:.0f}°", color=SLATE, fontsize=10)
    fig.tight_layout(); save(fig, P("fig04-05-agreement-cases.png"))

# 4.6 cosine and angle
def f6():
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    t = np.linspace(0, 180, 361); ax.plot(t, np.cos(np.radians(t)), color=PLUM)
    for a, b, dy in [("sat", "sat", 0.08), ("sat", "cat", 0.08), ("sat", "on", 0.1), ("on", "mat", 0.1)]:
        c = cosine(ROW[a], ROW[b]); g = np.degrees(np.arccos(np.clip(c, -1, 1)))
        ax.plot([g], [c], "o", color=NAVY, ms=8); ax.text(g + 3, c + dy, f"{a}/{b}: {g:.0f}°, {c:.3f}", color=NAVY, weight="bold")
    ax.set_xlabel("angle between two arrows (degrees)"); ax.set_ylabel("cosine")
    ax.set_xticks([0, 45, 90, 135, 180]); ax.set_ylim(-1.2, 1.25)
    ax.set_title("Cosine: agreement with the lengths taken out")
    save(fig, P("fig04-06-cosine-angle.png"))

# 4.7 one row of agreements
def f7():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, q in zip(axs, ["sat", "cat"]):
        v = [dot(ROW[q], ROW[w]) for w in SENT]
        ax.bar(range(6), v, color=[PLUM if x > 0 else (GREY if x == 0 else CORAL) for x in v])
        ax.set_xticks(range(6)); ax.set_xticklabels(SENT); ax.axhline(0, color=SLATE, lw=1)
        for i, x in enumerate(v): ax.text(i, x + (0.08 if x >= 0 else -0.25), f"{x:g}", ha="center", weight="bold", color=NAVY)
        ax.set_title(f"How much {q} agrees with each token")
    axs[0].set_ylabel("dot product")
    fig.tight_layout(); save(fig, P("fig04-07-one-row-agreements.png"))

# 4.8 length bias
def f8():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axs[0]; ax.set_xlim(-0.3, 2.5); ax.set_ylim(-0.3, 1.5); ax.set_aspect("equal")
    ax.axhline(0, color="#9aa5b1"); ax.axvline(0, color="#9aa5b1")
    vec(ax, ROW["a"], GREY, "a (2, 0)", lw=5, off=(-0.3, 0.1)); vec(ax, ROW["the"], AMBER, "the (1, 0)", off=(-0.4, -0.25))
    vec(ax, ROW["sat"], PLUM, "sat (1, 1)")
    ax.set_title("Same direction, different length")
    ax = axs[1]; x = np.arange(2)
    ax.bar(x - 0.18, [dot(ROW["sat"], ROW["the"]), dot(ROW["sat"], ROW["a"])], 0.36, color=PLUM, label="dot with sat")
    ax.bar(x + 0.18, [cosine(ROW["sat"], ROW["the"]), cosine(ROW["sat"], ROW["a"])], 0.36, color=BLUE, label="cosine with sat")
    ax.set_xticks(x); ax.set_xticklabels(["the", "a"]); ax.legend()
    ax.set_title("Dot rewards length; cosine ignores it")
    fig.tight_layout(); save(fig, P("fig04-08-length-bias.png"))

# 4.9 squash
def f9():
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.linspace(-6, 6, 300); ax.plot(x, squash(x), color=PLUM)
    for v in (-4, 0, 1, 4):
        ax.plot([v], [squash(v)], "o", color=NAVY, ms=7); ax.text(v + 0.2, squash(v) - 0.08, f"{v} -> {squash(v):.3f}", color=NAVY, weight="bold")
    ax.set_xlabel("agreement (dot product) of two rows"); ax.set_ylabel("how sure: neighbours?")
    ax.set_title("The squash turns any agreement into a number between 0 and 1")
    save(fig, P("fig04-09-squash.png"))

TRACK = [("said", "replied"), ("said", "tea"), ("queen", "king"), ("two", "three")]
A16, VOC, IDX, HIST = train(d=16, epochs=10, seed=1865, track=TRACK)

def f10():
    fig, ax = plt.subplots(figsize=(8.5, 4.6)); H = np.array(HIST)
    for k, (a, b) in enumerate(TRACK):
        ax.plot(range(len(H)), H[:, k], marker="o", color=[PLUM, GREY, NAVY, TEAL][k], label=f"{a} / {b}")
    ax.set_xlabel("passes through the novel"); ax.set_ylabel("cosine"); ax.legend(loc="lower right")
    ax.set_title("Nudging: words used in the same places drift together")
    save(fig, P("fig04-10-training-cosines.png"))

def f11():
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, q in zip(axs, ["said", "queen", "two"]):
        nb = nearest(A16, VOC, IDX, q, 6)[::-1]
        ax.barh([w for w, _ in nb], [c for _, c in nb], color=NAVY); ax.set_xlim(0.5, 1)
        for i, (_, c) in enumerate(nb): ax.text(c + 0.005, i, f"{c:.2f}", va="center", color=SLATE, fontsize=9)
        ax.set_title(f"nearest to '{q}'")
    fig.suptitle("Neighbours learned from 26,776 words in a second or two (16 numbers per word)", color=NAVY, weight="bold", x=0.02, ha="left")
    fig.tight_layout(); save(fig, P("fig04-11-neighbours.png"))

def f12():
    A2, V2, I2 = train(d=2, epochs=10, seed=1865)
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.scatter(A2[:, 0], A2[:, 1], s=8, color=GREY, alpha=0.5)
    for k, (w, c) in enumerate([("said", PLUM), ("replied", PLUM), ("tea", AMBER), ("queen", NAVY), ("mouse", TEAL)]):
        x, y = A2[I2[w]]; ax.scatter([x], [y], s=46, color=c, zorder=3)
        ax.annotate(w, (x, y), xytext=(1.6 + 0.0 * k, 2.4 - 0.3 * k), color=c, weight="bold",
                    arrowprops=dict(arrowstyle="-", color=c, lw=1))
    ax.axhline(0, color="#9aa5b1"); ax.axvline(0, color="#9aa5b1")
    ax.set_title("699 words in two numbers: all crowded onto one curve")
    ax.set_xlabel("first number"); ax.set_ylabel("second number")
    save(fig, P("fig04-12-two-numbers-crowd.png"))

# 4.13 analogy parallelogram
def f13():
    fig, ax = plt.subplots(figsize=(6.6, 5)); pts = {"man": (2, 1), "woman": (2, 3), "king": (5, 1), "queen": (5, 3)}
    ax.set_xlim(0, 6.2); ax.set_ylim(0, 4); ax.set_aspect("equal")
    for w, p in pts.items(): ax.plot(*p, "o", color=NAVY, ms=9); ax.text(p[0] + 0.1, p[1] + 0.12, f"{w} {p}", color=NAVY, weight="bold")
    vec(ax, (0, 2), PLUM, "woman - man", start=(2, 1), off=(-1.6, -1.1)); vec(ax, (0, 2), PLUM, None, start=(5, 1), ls="--")
    vec(ax, (3, 0), AMBER, "king - man", start=(2, 1), off=(-1.8, -0.4)); vec(ax, (3, 0), AMBER, None, start=(2, 3), ls="--")
    ax.set_title("king - man + woman lands on queen (hand-made rows)")
    save(fig, P("fig04-13-parallelogram.png"))

def rank_of(a, b, c, target):
    v = W[enc.encode(a)[0]] - W[enc.encode(b)[0]] + W[enc.encode(c)[0]]
    s = NW @ (v / np.linalg.norm(v)); order = np.argsort(-s)
    skip = {enc.encode(x)[0] for x in (a, b, c)}
    kept = [i for i in order[:3000] if i not in skip]
    t = enc.encode(target)[0]
    return kept.index(t) + 1 if t in kept else None

def f14():
    cases = [(" king", " man", " woman", " queen"), (" cats", " cat", " dog", " dogs"),
             (" Paris", " France", " Italy", " Rome"), (" walked", " walk", " swim", " swam")]
    ranks = [rank_of(*c) for c in cases]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = [f"{a.strip()} - {b.strip()} + {c.strip()} -> {t.strip()}" for a, b, c, t in cases]
    ax.barh(labels[::-1], ranks[::-1], color=[CORAL if r > 10 else NAVY for r in ranks[::-1]])
    ax.set_xscale("log"); ax.set_xlim(0.6, 1500); ax.set_xlabel("rank of the expected answer (1 = best; input words skipped)")
    for i, r in enumerate(ranks[::-1]): ax.text(r * 1.1, i, f"{r}", va="center", weight="bold", color=SLATE)
    ax.set_title("GPT-2's token table: some analogies hold, others do not")
    save(fig, P("fig04-14-gpt2-analogy-ranks.png"))

def f15():
    v = W[3797]; fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.bar(range(768), v, width=1.0, color=NAVY)
    big = np.argsort(-np.abs(v))[:3]
    for i in big: ax.text(i, v[i] + (0.02 if v[i] > 0 else -0.06), f"place {i}: {v[i]:.3f}", ha="center", fontsize=9, color=VERM, weight="bold")
    ax.set_xlim(-5, 773); ax.set_xlabel("place in the row (0 to 767)"); ax.set_ylabel("value")
    ax.set_title("GPT-2's row for ' cat' (id 3797): 768 small numbers, length 3.26")
    save(fig, P("fig04-15-gpt2-cat-row.png"))

def f16():
    q = NW[3797]; s = NW @ q; nb = [int(j) for j in np.argsort(-s)[1:9]]
    extra = [enc.encode(t)[0] for t in (" mat", " the")]
    ids = nb + extra; names = [repr(enc.decode([i])) for i in ids]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.barh(names[::-1], [s[i] for i in ids][::-1], color=[GREY] * 2 + [NAVY] * 8)
    for k, i in enumerate(ids[::-1]): ax.text(s[i] + 0.01, k, f"{s[i]:.2f}", va="center", fontsize=9)
    ax.set_xlabel("cosine with ' cat'"); ax.set_title("Nearest rows to ' cat' in GPT-2 (two far ones in grey)")
    save(fig, P("fig04-16-gpt2-cat-neighbours.png"))

def f17():
    L = np.linalg.norm(W, axis=1); fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.hist(L, bins=120, color=NAVY)
    for t, c in [(" the", AMBER), (" cat", BLUE), (" at", TEAL)]:
        x = L[enc.encode(t)[0]]; ax.axvline(x, color=c, lw=2); ax.text(x + 0.03, ax.get_ylim()[1] * 0.9, repr(t), color=c, weight="bold", rotation=90, va="top")
    ax.set_xlabel("row length"); ax.set_ylabel("number of rows")
    ax.set_title(f"Lengths of GPT-2's 50,257 rows: {L.min():.2f} to {L.max():.2f}, average {L.mean():.2f}")
    save(fig, P("fig04-17-gpt2-row-lengths.png"))

def f18():
    rng = np.random.default_rng(1865); fig, ax = plt.subplots(figsize=(9, 4.4))
    for d, c in [(2, CORAL), (16, AMBER), (128, NAVY), (768, PLUM)]:
        a, b = rng.normal(size=(5000, d)), rng.normal(size=(5000, d))
        cs = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
        ax.hist(cs, bins=np.linspace(-1, 1, 81), histtype="step", lw=2.2, color=c, label=f"{d} numbers", density=True)
    ax.set_xlabel("cosine between two random arrows"); ax.set_ylabel("share (density)"); ax.legend()
    ax.set_title("Wider rows: random arrows are nearly always at right angles")
    save(fig, P("fig04-18-random-angles.png"))

def f19():
    fig, ax = canvas(12, 5.2)
    label(ax, 1, 49, "Tying: one table reads tokens in and scores tokens out", NAVY, 12.5, ha="left", weight="bold")
    box(ax, 2, 22, 14, 9, "token id", AMBER, sub="e.g. 725 (␣cat)")
    box(ax, 40, 18, 30, 17, "the token table", NAVY, sub="2,048 rows x 128 numbers", fs=12.5)
    box(ax, 22, 22, 13, 9, "row 725", PLUM, sub="128 numbers")
    arrow(ax, 16.3, 26.5, 21.7, 26.5, AMBER); arrow(ax, 39.7, 30, 35.3, 28.5, NAVY)
    label(ax, 28, 18, "IN: look up one row", NAVY, 10, weight="bold")
    box(ax, 80, 36, 16, 8, "the model's work", BLUE, sub="Chapters 5 to 14", fs=10)
    arrow(ax, 28.5, 31.3, 80, 40, SLATE, rad=-0.25)
    box(ax, 80, 6, 16, 8, "final vector h", PLUM, sub="128 numbers", fs=10)
    arrow(ax, 88, 35.7, 88, 14.3, SLATE)
    box(ax, 101, 18, 17, 17, "2,048 scores", PLUM, sub="h . every row", fs=11)
    arrow(ax, 96.3, 10, 104, 17.7, PLUM); arrow(ax, 70.3, 22, 100.7, 22, NAVY, rad=0.15)
    label(ax, 86, 26, "OUT: agree with every row", NAVY, 10, weight="bold")
    label(ax, 60, 2, "untied, a second 2,048 x 128 table would do the OUT job: 262,144 more numbers", SLATE, 10)
    save(fig, P("fig04-19-tying.png"))

def f20():
    h = np.array((-1.2, 1.0)); s = [dot(h, r) for r in TOY_EMBED]
    order = np.argsort(s)[::-1]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axs[0]; plane(ax, 2.4)
    for w in ["hat", "rabbit", "mat", "at"]:
        vec(ax, ROW[w], GREY if w != "hat" else NAVY, w, lw=1.8)
    vec(ax, h, PLUM, "h (-1.2, 1)", lw=3.2, off=(-0.9, -0.45))
    ax.set_title("final vector h among the rows")
    ax = axs[1]
    ax.bar([TOY_VOCAB[i] for i in order], [s[i] for i in order], color=[PLUM if s[i] > 0 else CORAL for i in order])
    ax.axhline(0, color=SLATE, lw=1); ax.tick_params(axis="x", rotation=45)
    ax.set_title("score for every token = h . its row (same table)")
    fig.tight_layout(); save(fig, P("fig04-20-tied-scores.png"))

def f21():
    fig = plt.figure(figsize=(12, 5.2))
    ax = fig.add_axes([0.03, 0.08, 0.42, 0.82]); plane(ax, 1.6)
    for w in ["the", "cat", "sat", "on", "mat"]:
        vec(ax, ROW[w], COL[w], w, off=(0.06, -0.17) if w == "on" else (0.06, 0.06))
    t = np.linspace(np.radians(45), np.radians(90), 30); ax.plot(0.5 * np.cos(t), 0.5 * np.sin(t), color=NAVY, lw=2)
    ax.text(0.22, 0.6, "45°", color=NAVY, weight="bold")
    t = np.linspace(np.radians(-45), np.radians(45), 30); ax.plot(0.32 * np.cos(t), 0.32 * np.sin(t), color=GREY, lw=2)
    ax.text(0.36, -0.28, "sat to on: 90°", color=GREY, weight="bold")
    ax.set_title("SIGNATURE 2  Vectors as arrows, agreement as angle", fontsize=13)
    ax2 = fig.add_axes([0.5, 0.05, 0.48, 0.88]); ax2.axis("off"); ax2.set_xlim(0, 100); ax2.set_ylim(0, 100)
    lines = [("sat . cat", "1x0 + 1x1", "= 1", "cos 0.707", "45°", PLUM),
             ("sat . on", "1x1 + 1x(-1)", "= 0", "cos 0", "90°", GREY),
             ("on . mat", "1x(-1) + (-1)x1", "= -2", "cos -1", "180°", CORAL),
             ("sat . sat", "1x1 + 1x1", "= 2", "cos 1", "0°", NAVY)]
    ax2.text(0, 95, "multiply place by place, then add", color=NAVY, fontsize=14, weight="bold")
    for k, (a, b, c, d, e, col) in enumerate(lines):
        y = 80 - k * 15
        ax2.add_patch(FancyBboxPatch((0, y - 5), 98, 11, boxstyle="round,pad=0,rounding_size=2", fc=TINT.get(col, LIGHT), ec=col, lw=1.6))
        ax2.text(2, y, a, color=col, fontsize=14, weight="bold", va="center"); ax2.text(23, y, b, color=SLATE, fontsize=13, va="center")
        ax2.text(60, y, c, color=col, fontsize=14.5, weight="bold", va="center"); ax2.text(70, y, d, color=SLATE, fontsize=12.5, va="center")
        ax2.text(88, y, e, color=col, fontsize=13, weight="bold", va="center")
    ax2.text(0, 12, "cosine = dot / (length x length) = cos(angle)", color=NAVY, fontsize=14, weight="bold")
    ax2.text(0, 4, "every score in attention (Chapter 8) is one of these dot products", color=SLATE, fontsize=12)
    save(fig, P("fig04-21-signature-2.png"))

def f22():
    tok = load_bpe(os.path.join(ROOT, "04-code", "tiny", "tiny_bpe_2048.txt"))
    emb = Embedding(TINY_VOCAB_SIZE, 128, seed=1865); ids = tok.encode(" ".join(SENT)); X = emb(ids)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    im = ax.imshow(X, aspect="auto", cmap="RdBu_r", vmin=-0.06, vmax=0.06)
    ax.set_yticks(range(len(ids))); ax.set_yticklabels([f"{tok.piece(i)} ({i})" for i in ids])
    ax.set_xlabel("place in the row (0 to 127)"); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    ax.set_title("Tiny's sentence block before training: 7 rows of 128 small random numbers")
    save(fig, P("fig04-22-tiny-block.png"))

def f23():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axs[0]; plane(ax, 1.6)
    for w in ["the", "cat", "sat", "on", "mat"]: vec(ax, ROW[w], COL[w], w, lw=1.6, off=(0.06, -0.17) if w == "on" else (0.06, 0.06))
    ax.set_title("Before: five directions")
    ax = axs[1]; plane(ax, 1.6); vec(ax, (1, 1), CORAL, "every token (1, 1)", lw=3.5, off=(-1.2, 0.15))
    ax.set_title("Broken: one arrow, cosine 1 with everything", color=CORAL)
    fig.tight_layout(); save(fig, P("fig04-23-breakage-identical.png"))

def f24():
    m = W.mean(0) / np.linalg.norm(W.mean(0)); c = NW @ m; top = np.argsort(-c)[:100]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bins = np.linspace(c.min(), c.max(), 120)
    ax.hist(c, bins=bins, color=NAVY, label="all 50,257 rows"); ax.set_yscale("log")
    ax.hist(c[top], bins=bins, color=CORAL, label="the 100 closest: rarely seen tokens")
    x = c[3797]; ax.axvline(x, color=BLUE, lw=2); ax.text(x - 0.01, 300, "' cat'", color=BLUE, ha="right", weight="bold")
    ax.set_xlabel("cosine with the table's average direction"); ax.set_ylabel("number of rows (log scale)"); ax.set_ylim(0.6, 4e4); ax.legend(loc="upper left")
    ax.set_title("Every row leans one way; the rows training barely touched lean most")
    save(fig, P("fig04-24-untrained-rows.png"))

if __name__ == "__main__":
    for f in [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24]:
        f()
    print("chapter 4 figures:", len([n for n in os.listdir(OUT) if n.endswith(".png")]))
