"""Figures for Chapter 6 (Rotary positions), computed with the same code as the listings.
Run from the project root: python3 03-figures/src/ch06_figs.py"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code")); sys.path.insert(0, os.path.join(ROOT, "04-code", "ch04"))
sys.path.insert(0, os.path.join(ROOT, "04-code", "ch06"))
from style import *
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
import numpy as np
from tiny.tokenizer import TOY_VOCAB
from tiny.embed import (TOY_EMBED, TOY_TURN, clock, clock_speeds, add_positions, apply_rotary, agreement_table,
                        rotate_pair, dot, length, sinusoidal)
from bites import bf16
setup()
OUT = os.path.join(ROOT, "03-figures", "ch06"); os.makedirs(OUT, exist_ok=True)
P = lambda n: os.path.join(OUT, n)
SENT = "the cat sat on the mat".split()
SCR = "the mat sat on the cat".split()
ROW = {w: TOY_EMBED[i] for i, w in enumerate(TOY_VOCAB)}
COL = {"the": AMBER, "cat": BLUE, "sat": PLUM, "on": TEAL, "mat": VERM}
X = np.array([ROW[w] for w in SENT])
T30 = math.radians(30)
def turned(v, pos): return np.array(rotate_pair(v[0], v[1], pos * T30))

def plane(ax, lim=2.4):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, color="#9aa5b1", lw=1); ax.axvline(0, color="#9aa5b1", lw=1)
    ax.set_xlabel("first number (across)"); ax.set_ylabel("second number (up)")

def vec(ax, v, color, text=None, start=(0, 0), lw=2.6, off=(0.08, 0.08), fs=10.5, ls="-", alpha=1.0):
    ax.annotate("", xy=(start[0] + v[0], start[1] + v[1]), xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=15, ls=ls, alpha=alpha))
    if text:
        ax.text(start[0] + v[0] + off[0], start[1] + v[1] + off[1], text, color=color, fontsize=fs, weight="bold", alpha=alpha)

def heat(ax, T, cm, vmax, ttl, fs=10.5, color=NAVY):
    ax.grid(False); ax.imshow(T, cmap=cm, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(6), SENT, fontsize=8); ax.set_yticks(range(6), SENT, fontsize=8)
    for i in range(6):
        for j in range(6):
            v = round(float(T[i, j]), 2) + 0.0
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.62 * vmax else SLATE)
    ax.set_title(ttl, fontsize=fs, color=color)

def heading(ang):   # compass angle measured clockwise from 12 o'clock, in degrees
    return math.degrees(math.atan2(ang[0], ang[1])) % 360

# 6.1 absolute vs relative
def f1():
    fig, ax = canvas(12.5, 5.6)
    label(ax, 1, 53, "Absolute: each word gets a seat number.  Relative: what matters is how far apart two words sit",
          NAVY, 12, ha="left", weight="bold")
    for r, (start, y) in enumerate([(0, 33), (7, 12)]):
        label(ax, 1, y + 3, f"seats {start}-{start + 5}", GREY, 10, ha="left")
        for k, w in enumerate(SENT):
            box(ax, 14 + k * 17, y, 14, 7, w, COL[w], fs=12, sub=f"seat {start + k}")
        arrow(ax, 21, y + 8.5, 38, y + 8.5, CORAL, lw=2.2, rad=-0.4)
        label(ax, 29.5, y + 12.5, "gap 1", CORAL, 10, weight="bold")
        arrow(ax, 55, y - 1.0, 106, y - 1.0, TEAL, lw=2.0, rad=0.25)
        label(ax, 80, y - 5.2, "gap 3", TEAL, 10, weight="bold")
    label(ax, 117, 30, "seat numbers\nall changed", NAVY, 10.5, weight="bold")
    label(ax, 117, 18, "gaps\nunchanged", CORAL, 10.5, weight="bold")
    save(fig, P("fig06-01-absolute-relative.png"))

# 6.2 turning one arrow
def f2():
    fig, ax = plt.subplots(figsize=(6.4, 6.0)); plane(ax, 1.8)
    s = ROW["sat"]; t = turned(s, 2)
    ax.add_patch(Circle((0, 0), length(s), fill=False, ec=GREY, ls="--", lw=1))
    vec(ax, s, PLUM, "sat (1, 1)", off=(0.05, 0.05))
    vec(ax, t, CORAL, "turned 60\u00b0: (1.366, -0.366)", off=(-0.55, -0.22), lw=3)
    ax.add_patch(Arc((0, 0), 1.1, 1.1, theta1=-15, theta2=45, color=CORAL, lw=2))
    ax.text(0.66, 0.2, "60\u00b0\nclockwise", color=CORAL, fontsize=10, weight="bold")
    ax.text(-1.7, -1.6, "length 1.414 before and after:\nthe tip stays on the dashed circle", color=SLATE, fontsize=10)
    ax.set_title("Turning keeps the length and changes only the direction")
    save(fig, P("fig06-02-turn-one.png"))

# 6.3 the six rotated arrows
def f3():
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.6))
    for ax, words, ttl in zip(axs, (SENT, SCR), ("\u201cthe cat sat on the mat\u201d", "\u201cthe mat sat on the cat\u201d")):
        plane(ax, 1.9)
        for k, w in enumerate(words):
            v = turned(ROW[w], k)
            vec(ax, v, COL[w], f"{w}@{k}", off=(0.05, 0.05 if k % 2 == 0 else -0.2))
        ax.set_title(ttl)
    fig.suptitle("Each word turned 30\u00b0 per seat: the two 'the's now point different ways, and the two orders differ",
                 x=0.02, ha="left", color=NAVY, weight="bold", fontsize=12.5)
    fig.tight_layout(); save(fig, P("fig06-03-six-arrows.png"))

# 6.4 cat on the turntable = chapter 5's clock
def f4():
    fig, ax = plt.subplots(figsize=(6.2, 6.2)); plane(ax, 1.35)
    for k in range(12):
        v = turned(ROW["cat"], k)
        vec(ax, v, BLUE if k < 6 else GREY, str(k), off=(0.09 * v[0] - 0.04, 0.09 * v[1] - 0.03), lw=2 if k < 6 else 1.2)
    ax.set_title("cat (0, 1) turned for seats 0-11: the hands of Chapter 5's clock")
    save(fig, P("fig06-04-cat-clock.png"))

# 6.5 both turned vs one turned by the gap
def f5():
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.8))
    cases = [(0, 1, "the@0, cat@1"), (4, 5, "the@4, cat@5"), (None, 1, "the unturned, cat turned by the gap (1)")]
    for ax, (m, n, ttl) in zip(axs, cases):
        plane(ax, 1.3)
        a = ROW["the"] if m is None else turned(ROW["the"], m)
        b = turned(ROW["cat"], n - (0 if m is None else m)) if m is None else turned(ROW["cat"], n)
        vec(ax, a, AMBER, "the"); vec(ax, b, BLUE, "cat", off=(0.05, -0.18))
        ax.set_title(f"{ttl}\nagreement {dot(a, b):.3f}", fontsize=11)
    fig.suptitle("Turn both by the same amount and the angle between them stays put: only the gap counts",
                 x=0.02, ha="left", color=NAVY, weight="bold", fontsize=12.5)
    fig.tight_layout(); save(fig, P("fig06-05-both-vs-one.png"))

# 6.6 shift test: rotary vs added
def f6():
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 8.2))
    def rot(start): return agreement_table(apply_rotary(X, range(start, start + 6), speeds=TOY_TURN))
    def add(start): return agreement_table(add_positions(X, [clock(k) for k in range(start, start + 6)]))
    for r, (fn, name) in enumerate([(rot, "rotary (turned)"), (add, "Chapter 5 (added clock)")]):
        A, B = fn(0), fn(7)
        vmax = max(abs(A).max(), abs(B).max())
        for c, (T, ttl, cm) in enumerate([(A, "seats 0-5", "PuOr"), (B, "seats 7-12", "PuOr"), (B - A, "change", "RdBu")]):
            heat(axs[r, c], T, cm, vmax, f"{name}: {ttl}" if c < 2 else f"{name}: {int((abs(T) > 1e-9).sum())} of 36 change")
    fig.suptitle("Slide the sentence 7 seats along: rotary's table does not move at all; the added table does",
                 x=0.02, ha="left", color=NAVY, weight="bold", fontsize=12.5)
    fig.tight_layout(); save(fig, P("fig06-06-shift-test.png"))

# 6.7 score vs gap, formula
def f7():
    a, b = ROW["sat"]; c, d = ROW["on"]
    g = np.linspace(0, 24, 400); t = g * T30
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(g, (a * c + b * d) * np.cos(t) + (a * d - b * c) * np.sin(t), color=TEAL, label="formula: straight x cos + crossed x sin")
    gi = np.arange(0, 25)
    ax.plot(gi, [dot(ROW["sat"], turned(ROW["on"], k)) for k in gi], "o", color=CORAL, label="sat . (on turned by the gap), computed")
    ax.set_xlabel("gap (seats from sat to on)"); ax.set_ylabel("agreement")
    ax.set_title("sat and on: the agreement is a wave in the gap, repeating every 12 seats on the toy clock")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9.5); save(fig, P("fig06-07-score-vs-gap.png"))

# 6.8 four clocks for a row of eight
def f8():
    sp = clock_speeds(8); q = np.array([1.0, 0.0, 0.5, 0.5, -1.0, 1.0, 0.0, 2.0])
    fig, axs = plt.subplots(3, 4, figsize=(11, 8.4))
    for r, pos in enumerate([0, 1, 2]):
        v = apply_rotary([q], [pos], speeds=sp)[0]
        for c in range(4):
            ax = axs[r, c]; lim = 2.3; plane(ax, lim); ax.set_xlabel(""); ax.set_ylabel("")
            ax.tick_params(labelsize=7)
            vec(ax, q[2 * c:2 * c + 2], GREY, None, lw=1.2, ls="--")
            vec(ax, v[2 * c:2 * c + 2], SERIES[c], None)
            ax.set_title(f"clock {c} ({math.degrees(sp[c]):.2f}\u00b0/seat), seat {pos}", fontsize=9.5)
    fig.suptitle("A row of 8 numbers is 4 pairs, each on its own clock: fast clocks spin, slow clocks barely move",
                 x=0.02, ha="left", color=NAVY, weight="bold", fontsize=12.5)
    fig.tight_layout(); save(fig, P("fig06-08-four-clocks.png"))

# 6.9 Tiny's 64 clocks: turns in the window
def f9():
    sp = np.array(clock_speeds(128)); turns = sp * 1023 / (2 * math.pi)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(range(64), turns, color=[BLUE if t >= 1 else AMBER for t in turns])
    ax.set_yscale("log"); ax.axhline(1, color=CORAL, lw=1.6, ls="--")
    ax.text(40, 1.4, "one full turn", color=CORAL, fontsize=10, weight="bold")
    ax.text(38, 40, "36 clocks go all the way round\ninside 1,024 tokens (blue)", color=BLUE, fontsize=10, weight="bold")
    ax.text(38, 6, "28 never do (amber):\nclock 63 turns only 6.8\u00b0", color=AMBER, fontsize=10, weight="bold")
    ax.set_xlabel("clock number (pair of numbers)"); ax.set_ylabel("turns in 1,024 tokens (log scale)")
    ax.set_title("Tiny's 64 rotary clocks: from 163 turns down to a fiftieth of one")
    save(fig, P("fig06-09-tiny-turns.png"))

# 6.10 decay
def f10():
    d = 128; up = np.array([0.0, 1.0] * 64); gaps = np.unique(np.round(np.logspace(0, 3.3, 60)).astype(int))
    rng = np.random.default_rng(1865); Q = rng.normal(size=(150, d)); K = rng.normal(size=(150, d))
    def sc(a, b, g): return dot(apply_rotary([a], [0])[0], apply_rotary([b], [g])[0])
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(gaps, [sc(up, up, g) for g in gaps], color=NAVY, label="all hands at 12 o'clock (Chapter 5's curve)")
    ax.plot(gaps, [np.mean([sc(q, q, g) for q in Q]) for g in gaps], color=PLUM, label="same random row at both ends (average)")
    ax.plot(gaps, [np.mean([abs(sc(q, k, g)) for q, k in zip(Q, K)]) for g in gaps], color=AMBER, label="two different random rows (average size)")
    ax.set_xscale("log"); ax.set_xlabel("gap (tokens)"); ax.set_ylabel("agreement")
    ax.set_title("Fading with distance happens for rows that match, not for every pair")
    ax.legend(fontsize=9.5); save(fig, P("fig06-10-decay.png"))

# 6.11 scaling
def f11():
    d, s = 128, 4; plain = np.array(clock_speeds(d)); slowed = np.array(clock_speeds(d, 10000 * s ** (d / (d - 2))))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(range(64), np.full(64, 1 / s), color=TEAL, label="squeeze positions (every clock 4x slower)")
    ax.plot(range(64), slowed / plain, color=PLUM, label="change the base (fast clocks kept, slow clocks 4x slower)")
    ax.plot(range(64), np.ones(64), color=GREY, ls="--", label="do nothing")
    ax.set_xlabel("clock number"); ax.set_ylabel("speed after / speed before"); ax.set_ylim(0, 1.1)
    ax.set_title("Stretching Tiny from 1,024 to 4,096 tokens: two ways to slow the clocks")
    ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2); save(fig, P("fig06-11-scaling.png"))

# 6.12 bites
def f12():
    d = 128; rng = np.random.default_rng(1865); Q = rng.normal(size=(120, d)); K = rng.normal(size=(120, d))
    tr = clock_speeds(d); wrong = clock_speeds(d, 500000.0)
    order = [j for i in range(64) for j in (i, i + 64)]
    gaps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    def sc(q, k, g, sp=tr): return dot(apply_rotary([q], [0], speeds=sp)[0], apply_rotary([k], [g], speeds=sp)[0])
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for name, fn, col in [("served with base 500,000", lambda q, k, g: sc(q, k, g, wrong), PLUM),
                          ("pairs laid out as halves", lambda q, k, g: sc(q[order], k[order], g), AMBER),
                          ("clocks turning backwards", lambda q, k, g: sc(q, k, -g), CORAL)]:
        ax.plot(gaps, [np.mean([abs(fn(q, k, g) - sc(q, k, g)) for q, k in zip(Q, K)]) for g in gaps], "o-", color=col, label=name)
    ax.axhline(np.mean([abs(sc(q, k, 10)) for q, k in zip(Q, K)]), color=GREY, ls="--", label="typical size of an agreement")
    ax.set_xscale("log"); ax.set_xlabel("gap (tokens)"); ax.set_ylabel("average change in agreement")
    ax.set_title("Wrong rotary settings at serving time: nothing crashes, and the damage grows with the gap")
    ax.legend(fontsize=9.5); save(fig, P("fig06-12-bites.png"))

# 6.13 breakage
def f13():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
    good = lambda st: agreement_table(apply_rotary(X, range(st, st + 6), speeds=TOY_TURN))
    bad = lambda st: agreement_table(apply_rotary(X, range(st, st + 6), speeds=TOY_TURN), X)
    for ax, fn, ttl in [(axs[0], good, "both turned: change from seats 0-5 to 100-105"),
                        (axs[1], bad, "BROKEN, keys left unturned: same change")]:
        heat(ax, fn(100) - fn(0), "RdBu", 2.5, ttl, color=CORAL if "BROKEN" in ttl else NAVY)
    fig.suptitle("Turn only one side and the seat numbers leak back in: 32 of 36 agreements move",
                 x=0.02, ha="left", color=NAVY, weight="bold", fontsize=12.5)
    fig.tight_layout(); save(fig, P("fig06-13-breakage.png"))

# 6.14 bfloat16 positions
def f14():
    p = np.arange(990, 1031)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.step(p, p, where="mid", color=GREY, label="the true position")
    ax.step(p, bf16(p.astype(np.float32)), where="mid", color=CORAL, lw=2.6, label="the position after storing it in bfloat16")
    ax.set_xlabel("true position"); ax.set_ylabel("position used for the turn")
    ax.set_title("Store positions in bfloat16 and four neighbours share one turn")
    ax.legend(fontsize=9.5); save(fig, P("fig06-14-bf16.png"))

for f in (f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14):
    f()
print("chapter 6 figures written to", OUT)
