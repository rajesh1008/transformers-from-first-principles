"""Figures for Chapter 15, Workshop: Tracing the Sentence Through Tiny by Hand.  Every number is read from
04-code/ch15/outputs (trace.json, check.json) or recomputed here from 04-code/ch15/hand.py and pencil.py.
Run from the project root: python3 03-figures/src/ch15_figs.py"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code")); sys.path.insert(0, os.path.join(ROOT, "04-code", "ch15"))
from style import *
import matplotlib.pyplot as plt
setup()
OUT = os.path.join(ROOT, "03-figures", "ch15"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch15", "outputs")
js = lambda n: json.load(open(os.path.join(CH, n + ".json")))
T = js("trace"); C = js("check")
from hand import WORDS, IDS, SEAT_WORDS, hand_tiny
from tiny.embed import TOY_EMBED, TOY_POS
from tiny.norm import layer_norm, rms_norm
SEATS = [f"{w}@{i}" for i, w in enumerate(SEAT_WORDS)]
x0 = np.array(T["x0"]); L0 = T["layer0"]; L1 = T["layer1"]

# 01 the plan: the whole hand trace as a stack, with the shape written down at each step
fig, ax = canvas(7.2, 10.2)
stages = [("token rows + seat rows", AMBER, "6 rows of 2"), ("layer 0: ln_1, one head, add", NAVY, "shares 6 x 6"),
          ("layer 0: ln_2, feed-forward 2 -> 3 -> 2, add", NAVY, "6 rows of 2"),
          ("layer 1: ln_1, one head, add", NAVY, "shares 6 x 6"), ("layer 1: ln_2, feed-forward, add", NAVY, "6 rows of 2"),
          ("final norm (ln_f)", TEAL, "6 rows of 2"), ("scores against the token table", PLUM, "6 x 12"),
          ("shares", PLUM, "6 x 12, each row sums to one")]
ys = [3 + 11.5 * k for k in range(len(stages))]
for (name, c, sub), y in zip(stages, ys):
    box(ax, 8, y, 56, 8, name, color=c, fs=10.5, sub=sub)
for a, b in zip(ys[:-1], ys[1:]):
    arrow(ax, 36, a + 8, 36, b)
label(ax, 36, 99.5, "The hand trace: 112 learned numbers, 624 numbers written down", color=NAVY, fs=11, weight="bold")
save(fig, os.path.join(OUT, "fig15-01-plan.png"))

# 02 tokens to rows: token rows (dashed) and token + seat rows (solid)
fig, ax = plt.subplots(figsize=(7.2, 7.2))
for i in range(6):
    t = TOY_EMBED[IDS[i]]; s = x0[i]
    ax.annotate("", xy=t, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2, ls="--"))
    ax.annotate("", xy=s, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.4))
    ax.annotate("", xy=s, xytext=t, arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.2))
    ax.text(s[0] + 0.05, s[1] + 0.06, f"{SEATS[i]} ({s[0]:g}, {s[1]:g})", color=NAVY, fontsize=10)
ax.set_xlim(-1.6, 2.2); ax.set_ylim(-1.4, 2.0); ax.set_aspect("equal")
ax.axhline(0, color="#9aa5b1", lw=0.8); ax.axvline(0, color="#9aa5b1", lw=0.8)
ax.set_xlabel("first number"); ax.set_ylabel("second number")
ax.set_title("Token row (dashed grey) + seat row (thin teal) = row in (amber)")
save(fig, os.path.join(OUT, "fig15-02-tokens-to-rows.png"))

# 03 one head: layer 0 share table (Signature 3)
A = np.array(L0["shares"])
fig, ax = plt.subplots(figsize=(7.2, 6.4))
M = np.ma.masked_where(np.triu(np.ones((6, 6)), 1) > 0, A)
cmap = plt.cm.Purples.copy(); cmap.set_bad("#f2f2f2")
ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{A[i, j]:.3f}" if j <= i else "masked", ha="center", va="center", fontsize=9.5,
                color="white" if (j <= i and A[i, j] > 0.55) else (SLATE if j <= i else GREY))
ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.set_xticklabels(SEATS, rotation=30); ax.set_yticklabels(SEATS)
ax.set_xlabel("seat being read (key)"); ax.set_ylabel("seat reading (query)"); ax.grid(False)
ax.set_title("Layer 0's Attention Table for the running sentence")
save(fig, os.path.join(OUT, "fig15-03-attention-table.png"))

# 04 one block: sat's row through layer 0
s = 2
pts = {"row in": x0[s], "ln_1": np.array(L0["ln_1"][s]), "add_1": np.array(L0["add_1"][s]), "ln_2": np.array(L0["ln_2"][s]), "row out": np.array(L0["add_2"][s])}
fig, ax = plt.subplots(figsize=(7.2, 6.0))
th = np.linspace(0, 2 * np.pi, 400); ax.plot(np.sqrt(2) * np.cos(th), np.sqrt(2) * np.sin(th), color=TEAL, lw=1.0, ls=":")
ax.text(-1.45, -1.25, "rows of RMS 1 (length 1.414)", color=TEAL, fontsize=9)
ax.annotate("", xy=pts["add_1"], xytext=pts["row in"], arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2.4))
ax.annotate("", xy=pts["row out"], xytext=pts["add_1"], arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2.4, ls="--"))
for k, c, mk in (("row in", AMBER, "o"), ("ln_1", TEAL, "s"), ("add_1", PLUM, "o"), ("ln_2", TEAL, "s"), ("row out", PLUM, "D")):
    p = pts[k]; ax.plot(*p, marker=mk, color=c, ms=9)
    off = {"row in": (-0.1, -0.28, "right"), "ln_1": (-0.12, 0.2, "right"), "ln_2": (0.1, -0.3, "left"), "add_1": (0.1, -0.2, "left"), "row out": (0.1, 0.08, "left")}[k]
    ax.text(p[0] + off[0], p[1] + off[1], f"{k} ({p[0]:.3f}, {p[1]:.3f})", fontsize=9.5, color=SLATE, ha=off[2])
ax.text(1.55, 0.95, "attention write\n(1.863, 0.792)", color=NAVY, fontsize=9.5)
ax.text(3.3, 1.5, "feed-forward write\n(0.291, 0.645)", color=NAVY, fontsize=9.5)
ax.set_xlim(-1.6, 4.2); ax.set_ylim(-1.5, 2.4); ax.set_aspect("equal")
ax.set_xlabel("first number"); ax.set_ylabel("second number")
ax.set_title("Sat@2 through layer 0: two writes added to the kept row")
save(fig, os.path.join(OUT, "fig15-04-one-block.png"))

# 05 two layers: every row's path
fig, ax = plt.subplots(figsize=(7.2, 6.4))
a2 = np.array(L0["add_2"]); top = np.array(T["top"])
mk = ["o", "s", "^", "v", "D", "P"]
for i in range(6):
    ax.plot([x0[i, 0], a2[i, 0], top[i, 0]], [x0[i, 1], a2[i, 1], top[i, 1]], color=SERIES[i % len(SERIES)], lw=1.6, marker=mk[i], ms=7)
    dy = {0: -0.32, 2: -0.22, 4: 0.14}.get(i, 0.12)
    ax.text(x0[i, 0] + (0.15 if i == 0 else -0.05), x0[i, 1] + dy, SEATS[i], fontsize=9.5, color=SLATE, ha="left" if i == 0 else "right")
    ax.text(top[i, 0] + 0.08, top[i, 1], f"{SEATS[i]} ({top[i, 0]:.2f}, {top[i, 1]:.2f})", fontsize=8.8, color=SLATE)
ax.set_xlim(-1.5, 7.6); ax.set_ylim(-1.3, 4.0)
ax.set_xlabel("first number"); ax.set_ylabel("second number")
ax.set_title("Rows entering layer 0, after layer 0 and after layer 1")
ax.text(-1.3, 3.6, "token similarity 0.300 -> 0.890 -> 0.967", color=NAVY, fontsize=10)
save(fig, os.path.join(OUT, "fig15-05-two-layers.png"))

# 06 the output distribution
sh = np.array(T["out_shares"])
fig, (a1, a2_) = plt.subplots(2, 1, figsize=(7.2, 8.4), gridspec_kw={"height_ratios": [3, 2]})
order = np.argsort(-sh[5])
cols = [PLUM if k != order[0] else NAVY for k in order]
a1.barh(range(12), sh[5, order][::-1], color=cols[::-1])
a1.set_yticks(range(12)); a1.set_yticklabels([WORDS[k] for k in order][::-1])
for y, v in enumerate(sh[5, order][::-1]):
    a1.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9, color=SLATE)
a1.axvline(1 / 12, color=GREY, ls="--", lw=1); a1.text(1 / 12 + 0.004, 11.4, "even split 1/12", fontsize=9, color=GREY)
a1.set_xlabel("share"); a1.set_title("Shares for the word after mat@5")
true = [sh[i, IDS[i + 1]] for i in range(5)]
a2_.bar(range(5), true, color=AMBER, hatch="//", edgecolor=SLATE)
for x, v in enumerate(true):
    a2_.text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=9, color=SLATE)
a2_.axhline(1 / 12, color=GREY, ls="--", lw=1)
a2_.set_xticks(range(5)); a2_.set_xticklabels([f"{SEATS[i]}\n-> {WORDS[IDS[i + 1]]}" for i in range(5)], fontsize=9)
a2_.set_ylabel("share"); a2_.set_title("Share given to the true next word")
fig.tight_layout(); save(fig, os.path.join(OUT, "fig15-06-output.png"))

# 07 checking against the code
names = [f"{r[0]}" + ("" if r[1] is None else f" L{r[1]}") for r in C["rows_eps"]]
fig, ax = plt.subplots(figsize=(7.2, 7.4))
y = np.arange(len(names))
for key, c, m_, lab in (("rows_rounded", VERM, "o", "pencil rounded to 3 places"), ("rows_eps", PLUM, "s", "code with eps 1e-5"),
                        ("rows_eps0", TEAL, "^", "code with eps 0")):
    v = np.array([max(r[2], 1e-18) for r in C[key]])
    ax.plot(v, y, marker=m_, color=c, lw=1.2, label=lab)
ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9); ax.invert_yaxis()
ax.set_xlabel("largest difference from the pencil trace (log scale)"); ax.legend(loc="lower right", fontsize=9)
ax.set_title("Every step, checked against Tiny's code")
save(fig, os.path.join(OUT, "fig15-07-check.png"))

# 08 width 2: layer norm against RMSNorm, and neuron 3's line
fig, ax = plt.subplots(figsize=(7.2, 7.0))
ax.plot(np.sqrt(2) * np.cos(th), np.sqrt(2) * np.sin(th), color=TEAL, lw=1.0, ls=":")
for i in range(6):
    r = x0[i]; ln = layer_norm(r); rn = rms_norm(r)
    ax.plot(*r, "o", color=AMBER, ms=7); ax.plot(*rn, "s", color=TEAL, ms=7); ax.plot(*ln, "x", color=VERM, ms=10, mew=2.2)
    ax.annotate("", xy=rn, xytext=r, arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.0))
    ax.text(r[0] + 0.06, r[1] + (-0.17 if i == 2 else 0.06), SEATS[i], fontsize=9, color=SLATE)
xx = np.linspace(-0.4, 2.4, 10); ax.plot(xx, 2 - xx, color=NAVY, ls="--", lw=1.5)
ax.text(1.55, 0.25, "neuron 3 fires only\nabove x1 + x2 = 2", color=NAVY, fontsize=9.5)
ax.text(-1.55, -1.25, "o rows in   s RMSNorm   x layer norm", color=SLATE, fontsize=9.5)
ax.set_xlim(-1.7, 2.4); ax.set_ylim(-1.5, 2.1); ax.set_aspect("equal")
ax.set_xlabel("first number"); ax.set_ylabel("second number")
ax.set_title("Two numbers: layer norm keeps a sign, RMSNorm keeps a direction")
save(fig, os.path.join(OUT, "fig15-08-width-two.png"))

# 09 Lab 15 breakage 1: where cat's change travels
from tiny.attention import softmax
m = hand_tiny(); E1 = TOY_EMBED.copy(); E1[1, 1] = 1.5; m1 = hand_tiny(E=E1)
def steps(m, ids):
    X = m.embed(ids); out = [("rows in", X)]
    for l, b in enumerate(m.blocks):
        st = b.explode(X); _, s_ = b.att(b.ln_1(X), return_shares=True)
        out += [(f"L{l} ln_1", st["ln_1"]), (f"L{l} shares", s_[0]), (f"L{l} attention", st["attention"]), (f"L{l} add_1", st["add_1"]),
                (f"L{l} ln_2", st["ln_2"]), (f"L{l} feed-forward", st["feed_forward"]), (f"L{l} add_2", st["add_2"])]
        X = st["add_2"]
    S = m.forward(ids); out += [("ln_f", m.top_rows(X)), ("scores", S), ("shares", softmax(S))]
    return out
a, b = steps(m, IDS), steps(m1, IDS)
D = np.array([np.abs(np.asarray(y_) - np.asarray(x_)).max(axis=1) for (_, x_), (_, y_) in zip(a, b)])
fig, ax = plt.subplots(figsize=(7.2, 8.4))
Dl = np.log10(np.maximum(D, 1e-6))
im = ax.imshow(Dl, cmap="Oranges", vmin=-4, vmax=0, aspect="auto")
for i in range(D.shape[0]):
    for j in range(6):
        ax.text(j, i, "0" if D[i, j] == 0 else f"{D[i, j]:.3f}", ha="center", va="center", fontsize=8.5,
                color=GREY if D[i, j] == 0 else ("white" if Dl[i, j] > -1.2 else SLATE))
ax.set_xticks(range(6)); ax.set_xticklabels(SEATS, rotation=30); ax.set_yticks(range(len(a))); ax.set_yticklabels([n for n, _ in a], fontsize=9)
ax.grid(False); ax.set_title("Breakage 1: cat's row (0, 1) -> (0, 1.5), largest change per seat")
save(fig, os.path.join(OUT, "fig15-09-lab-breakage.png"))
print("wrote", sorted(os.listdir(OUT)))
