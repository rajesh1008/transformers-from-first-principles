"""Figures for Chapter 9, Many Heads.  Every number is computed here from the chapter's own code
(04-code/ch09, seed 1865) or read from the outputs it saved.
Run from the project root: python3 03-figures/src/ch09_figs.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
setup()
OUT = os.path.join(ROOT, "03-figures", "ch09"); os.makedirs(OUT, exist_ok=True)
CODE = os.path.join(ROOT, "04-code"); CH9 = os.path.join(CODE, "ch09")
sys.path.insert(0, CODE); sys.path.insert(0, CH9)
from tiny.attention import attention, softmax, causal_mask
from tiny.embed import TOY_POS
words = ["the", "cat", "sat", "on", "the", "mat"]
X = np.array([[1, 0], [0, 1], [1, 1], [1, -1], [1, 0], [-1, 1]], dtype=float)
W_q = np.array([[1, 0], [0, 2]], float); W_k = np.array([[0, 1], [1, 0]], float); W_v = np.array([[1, 0], [1, 1]], float)
plum_map = LinearSegmentedColormap.from_list("sh", ["#ffffff", "#efe7f5", PLUM])
M6 = causal_mask(6)

def heat_ax(ax, A, title_, rows=words, cols=words, fs=9.5, mask=True, hl=None, fmt="{:.2f}"):
    n = A.shape[0]
    ax.imshow(A, cmap=plum_map, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if mask and j > i:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc="#f3f4f6", ec="#c7ccd4", hatch="///", lw=0.4))
            elif fs:
                ax.text(j, i, fmt.format(A[i, j]), ha="center", va="center", fontsize=fs, color=NAVY)
    if hl:
        for (i, j) in hl:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=CORAL, lw=2.2))
    ax.set_xticks(range(n), cols, fontsize=9); ax.set_yticks(range(n), rows, fontsize=9)
    ax.grid(False); ax.set_title(title_, fontsize=11.5)

# 01 one blend is not enough: mat's shares under head A, head B and a single head asked for both
rows = [("head A: wants sat", [0, 0, 1, 0, 0, 0], (1, 1), BLUE),
        ("head B: wants on", [0, 0, 0, 1, 0, 0], (1, -1), TEAL),
        ("one head for both", [0.25, 0, 0.25, 0.25, 0.25, 0], (1, 0), CORAL)]
fig, axs = plt.subplots(3, 1, figsize=(7.6, 7.4), sharex=True)
for ax, (t, sh, out, col) in zip(axs, rows):
    ax.bar(range(6), sh, color=TINT.get(col, LIGHT), edgecolor=col, lw=1.8, hatch="" if col != CORAL else "//")
    for k, v in enumerate(sh):
        ax.text(k, v + 0.04, f"{v:.2f}", ha="center", fontsize=9.5, color=NAVY)
    ax.set_ylim(0, 1.25); ax.set_ylabel("mat's share")
    ax.set_title(f"{t}: output {out}", fontsize=11.5)
axs[2].set_xticks(range(6), words)
axs[2].text(5.4, 0.85, "(1, 0) is the value\nof 'the', not a mix\nanyone can read", ha="right", va="top", fontsize=9.5, color=CORAL)
save(fig, os.path.join(OUT, "fig09-01-one-blend.png"))

# 02 split and join: Signature 3 side by side
fig, ax = canvas(9, 7.6)
title(ax, "Many heads: cut the width, run each head, join, mix", x=1, y=74)
box(ax, 2, 33, 12, 12, "X", NAVY, sub="6 x 128")
for h in range(4):
    y = 58 - h * 15
    box(ax, 22, y, 16, 9, f"head {h + 1}", BLUE, sub="Q, K, V: 128 x 32", fs=10.5)
    arrow(ax, 14, 39, 22, y + 4.5, color=SLATE, lw=1.4)
    box(ax, 44, y, 14, 9, f"blend {h + 1}", PLUM, sub="6 x 32", fs=10.5)
    arrow(ax, 38, y + 4.5, 44, y + 4.5, color=SLATE, lw=1.4)
    arrow(ax, 58, y + 4.5, 63, 39, color=SLATE, lw=1.4)
box(ax, 63, 33, 11, 12, "side by\nside", AMBER, fs=10, sub=None)
label(ax, 68.5, 30, "6 x 128", fs=9)
arrow(ax, 74, 39, 78, 39, color=SLATE, lw=1.8)
box(ax, 78, 33, 11, 12, "W_o", NAVY, sub="128 x 128", fs=10.5)
label(ax, 47, 2.5, "each head: its own score table, mask, softmax and blend (Signature 3)", fs=9.5, color=GREY)
save(fig, os.path.join(OUT, "fig09-02-split-join.png"))

# 03 two heads of width 1 cut from Chapter 8's head
Q, K, V = X @ W_q, X @ W_k, X @ W_v
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.6))
_, A1 = attention(X, W_q, W_k, W_v)
heat_ax(axs[0], A1, "Chapter 8: one head, width 2", hl=[(2, 1)])
for h in (0, 1):
    A = softmax(np.where(M6, Q[:, h:h + 1] @ K[:, h:h + 1].T, -np.inf))
    heat_ax(axs[h + 1], A, f"head {h + 1}: column {h + 1} only, width 1", hl=[(2, 1)])
save(fig, os.path.join(OUT, "fig09-03-two-narrow-heads.png"))

# 04 Interlude: side by side times W_o equals the sum of the blocks
fig, ax = canvas(9, 5.4)
title(ax, "Cutting the output table into blocks", x=1, y=52)
box(ax, 2, 28, 10, 8, "[1, 2", BLUE, fs=11); box(ax, 12, 28, 10, 8, "3, -1]", TEAL, fs=11)
label(ax, 12, 24, "head 1 | head 2", fs=9)
label(ax, 25, 32, "x", fs=14)
box(ax, 28, 34, 22, 8, "rows 1-2 of W_o", BLUE, fs=10); box(ax, 28, 24, 22, 8, "rows 3-4 of W_o", TEAL, fs=10)
label(ax, 54, 32, "=", fs=14)
box(ax, 58, 28, 14, 8, "[2, 5, 5]", PLUM, fs=11)
label(ax, 45, 13, "[1, 2] x rows 1-2  =  [1, 2, 4]", color=BLUE, fs=11)
label(ax, 45, 8, "[3, -1] x rows 3-4 =  [1, 3, 1]", color=TEAL, fs=11)
label(ax, 45, 3, "sum                =  [2, 5, 5]", color=PLUM, fs=11)
save(fig, os.path.join(OUT, "fig09-04-blocks.png"))

# 05 how many heads: time and shares per window
import json
t = json.load(open(os.path.join(CH9, "outputs", "how_many_timing.json")))
hs = [1, 2, 4, 8, 16, 32, 64, 128]
fig, axs = plt.subplots(2, 1, figsize=(7.4, 7.4), sharex=True)
axs[0].plot(range(len(hs)), [t[str(h)] for h in hs], marker="o", color=BLUE)
for k, h in enumerate(hs):
    axs[0].text(k, t[str(h)] * 1.25, f"{t[str(h)]:.0f}", ha="center", fontsize=9, color=NAVY)
axs[0].set_ylabel("ms per 1,024-token window"); axs[0].set_yscale("log"); axs[1].set_yscale("log"); axs[0].set_title("Same learned numbers (65,536), more heads, more time", fontsize=11.5)
axs[1].bar(range(len(hs)), [h * 1.048576 for h in hs], color=TINT[PLUM], edgecolor=PLUM, hatch="..")
for k, h in enumerate(hs):
    axs[1].text(k, h * 1.048576 * 1.12, f"{h * 1.048576:.1f} M", ha="center", fontsize=9, color=NAVY)
axs[1].set_ylabel("shares per window (millions)"); axs[1].set_xticks(range(len(hs)), [f"{h} x {128 // h}" for h in hs])
axs[1].set_xlabel("heads x width of each head (Tiny, width 128)")
save(fig, os.path.join(OUT, "fig09-05-how-many.png"))

# 06 the hand-built induction pair on the repeated sentence
import importlib.util
spec = importlib.util.spec_from_file_location("ind", os.path.join(CH9, "induction_toy.py"))
seq = [0] + [1, 2, 3, 4, 1, 5] * 2; vocab = ["<s>", "the", "cat", "sat", "on", "mat"]
names = [vocab[t] for t in seq]; Vn, n = 6, len(seq)
src = open(os.path.join(CH9, "induction_toy.py")).read()
ns = {}
exec(src.split('vocab = ["<s>"')[0], ns)
R0 = ns["build_rows"](seq, Vn); w1, A1i = ns["look_back"](R0, Vn, n, 1); out2, A2i = ns["lookup"](R0 + w1, Vn, n)
fig, axs = plt.subplots(1, 2, figsize=(13, 6.2))
heat_ax(axs[0], A1i, "layer 1: previous-token head", rows=names, cols=names, fs=0)
heat_ax(axs[1], A2i, "layer 2: lookup head reads layer 1", rows=names, cols=names, fs=0, hl=[(8, 3), (9, 4), (10, 5)])
save(fig, os.path.join(OUT, "fig09-06-induction-pair.png"))

# 07-10 GPT-2 circuits
d = np.load(os.path.join(CH9, "outputs", "gpt2_circuits.npz"))
fig, axs = plt.subplots(1, 2, figsize=(12.5, 6))
for ax, A, t_ in ((axs[0], d["S_prev"], "head 4.11: previous-token"), (axs[1], d["S_ind"], "head 5.5: induction")):
    ax.imshow(A, cmap=plum_map, vmin=0, vmax=1); ax.grid(False)
    ax.set_title(t_ + " (101 tokens)", fontsize=11.5); ax.set_xlabel("seat looked at"); ax.set_ylabel("seat looking")
    ax.axhline(50.5, color=GREY, lw=0.8, ls="--"); ax.axvline(50.5, color=GREY, lw=0.8, ls="--")
axs[1].text(60, 20, "second copy looks at\nthe seat after its\nearlier copy", fontsize=9.5, color=CORAL)
save(fig, os.path.join(OUT, "fig09-07-gpt2-stripes.png"))

fig, ax = plt.subplots(figsize=(7.2, 6.4))
ind, prev = d["induction"].ravel(), d["prev"].ravel()
layers = np.repeat(np.arange(12), 12)
sc = ax.scatter(prev, ind, c=layers, cmap=LinearSegmentedColormap.from_list("l", [AMBER, BLUE, NAVY]), s=36, edgecolor="white", lw=0.5)
for i in np.argsort(-ind)[:5]:
    ax.annotate(f"{i // 12}.{i % 12}", (prev[i], ind[i]), xytext=(6, -3), textcoords="offset points", fontsize=9, color=NAVY)
ax.annotate("4.11", (prev[4 * 12 + 11], ind[4 * 12 + 11]), xytext=(-30, 8), textcoords="offset points", fontsize=9, color=NAVY)
ax.set_xlabel("previous-token score"); ax.set_ylabel("induction score")
ax.set_title("GPT-2's 144 heads: two jobs, different heads", fontsize=11.5)
cb = fig.colorbar(sc, ax=ax, fraction=0.04); cb.set_label("layer")
save(fig, os.path.join(OUT, "fig09-08-head-scores.png"))

loss = d["loss"]
fig, ax = plt.subplots(figsize=(8.4, 5.2))
ax.plot(np.arange(1, 101), loss, color=BLUE, lw=1.6)
ax.axvline(50.5, color=GREY, ls="--", lw=1)
ax.text(25, 14.2, "first copy: mean 12.912", ha="center", fontsize=10, color=NAVY)
ax.text(76, 3.0, "second copy: mean 0.207", ha="center", fontsize=10, color=NAVY)
ax.axhline(np.log(50257), color=GREY, ls=":", lw=1); ax.text(99, np.log(50257) + 0.3, "uniform guess 10.825", ha="right", fontsize=9, color=GREY)
ax.set_xlabel("seat (predicting the next token)"); ax.set_ylabel("loss (natural log)")
ax.set_title("GPT-2 small on 50 random tokens shown twice", fontsize=11.5)
save(fig, os.path.join(OUT, "fig09-09-repeat-loss.png"))

fig, ax = plt.subplots(figsize=(6.8, 6.4))
ez, en = d["eff_zero"].ravel(), d["eff_naive"].ravel()
ax.scatter(ez, en, s=30, color=TINT[BLUE], edgecolor=BLUE)
ax.set_xlabel("loss change, head removed before W_o (correct)"); ax.set_ylabel("loss change, 64 columns zeroed after W_o")
ax.set_title("Two ways to remove a head: correlation 0.07", fontsize=11.5)
for i in np.argsort(-ez)[:3]:
    ax.annotate(f"{i // 12}.{i % 12}", (ez[i], en[i]), xytext=(5, 4), textcoords="offset points", fontsize=9, color=NAVY)
save(fig, os.path.join(OUT, "fig09-10-two-removals.png"))

# 11 pruning
p = os.path.join(CH9, "outputs", "gpt2_prune.npz")
if os.path.exists(p):
    q = np.load(p); eff = np.sort(q["eff"].ravel()); rows_ = q["rows"]; base = q["base"]
    fig, axs = plt.subplots(2, 1, figsize=(7.6, 8.6)); fig.subplots_adjust(hspace=0.42)
    axs[0].bar(range(144), eff, color=[TINT[CORAL] if e > 0.01 else TINT[GREY] for e in eff], edgecolor=[CORAL if e > 0.01 else GREY for e in eff], lw=0.6)
    axs[0].axhline(0.01, color=GREY, ls=":", lw=1); axs[0].axhline(-0.01, color=GREY, ls=":", lw=1)
    axs[0].set_ylabel("loss change on Alice"); axs[0].set_xlabel("the 144 heads, removed one at a time, sorted")
    axs[0].set_title("Most heads matter little on their own", fontsize=11.5)
    ks = [0] + [int(r[0]) for r in rows_]
    axs[1].plot(ks, [base[1]] + [r[2] for r in rows_], marker="o", color=BLUE, label="least useful heads (chosen on Alice)")
    axs[1].plot(ks, [base[1]] + [r[3] for r in rows_], marker="s", ls="--", color=CORAL, label="random heads (mean of 5)")
    axs[1].set_xlabel("heads removed together"); axs[1].set_ylabel("loss on Looking-Glass")
    axs[1].legend(fontsize=9.5); axs[1].set_title("Which heads you remove matters more than how many", fontsize=11.5)
    save(fig, os.path.join(OUT, "fig09-11-pruning.png"))

# 12 gallery: GPT-2's heads on the running sentence by kind, per layer
S = np.load(os.path.join(CODE, "ch08", "outputs", "gpt2_shares.npy")); r = np.arange(1, 6)
pv = S[:, :, r, r - 1].mean(-1); fs_ = S[:, :, r, 0].mean(-1); sf = S[:, :, r, r].mean(-1)
lab = np.full((12, 12), 3); lab[sf > 0.5] = 2; lab[fs_ > 0.5] = 0; lab[pv > 0.5] = 1
fig, ax = plt.subplots(figsize=(8.4, 5.4))
bottom = np.zeros(12)
for k, (nm, col, hatch) in enumerate([("first token", GREY, ""), ("previous token", BLUE, "//"), ("self", TEAL, ".."), ("mixed", AMBER, "xx")]):
    c = (lab == k).sum(1)
    ax.bar(range(12), c, bottom=bottom, color=TINT.get(col, LIGHT), edgecolor=col, hatch=hatch, label=nm, lw=1)
    bottom += c
ax.set_xticks(range(12)); ax.set_xlabel("layer"); ax.set_ylabel("heads")
ax.legend(ncol=4, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.14))
ax.set_title("GPT-2's heads on the running sentence, by what rows 1-5 favour", fontsize=11.5)
save(fig, os.path.join(OUT, "fig09-12-gallery.png"))

# 13 Lab 9 checkpoint: four tables
rng = np.random.default_rng(1865)
heads = [tuple(np.round(rng.normal(size=(2, 2)), 2) for _ in range(3)) for _ in range(4)]
fig, axs = plt.subplots(2, 2, figsize=(10.5, 9.4))
for h, ax in enumerate(axs.ravel()):
    _, A = attention(X + TOY_POS, *heads[h])
    heat_ax(ax, A, f"Lab 9, head {h}", fs=8.5, hl=[(2, int(np.argmax(A[2])))])
save(fig, os.path.join(OUT, "fig09-13-lab-checkpoint.png"))

# 14 the width-1 head: every row's favourite is the running largest or smallest key
rng = np.random.default_rng(1865)
R = rng.normal(size=(1024, 128))
wq, wk, wv = (rng.normal(size=(128, 1)) / np.sqrt(128) for _ in range(3))
_, A = attention(R, wq, wk, wv)
k = (R @ wk)[:, 0]; fav = A.argmax(1)
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.scatter(range(1024), k, s=5, color=GREY, alpha=0.5, label="each token's key (one number)")
ax.plot(np.maximum.accumulate(k), color=BLUE, lw=1.6, label="largest key so far")
ax.plot(np.minimum.accumulate(k), color=TEAL, lw=1.6, label="smallest key so far")
fs = sorted(set(fav.tolist()))
ax.scatter(fs, k[fs], s=60, facecolor="none", edgecolor=CORAL, lw=1.8, label=f"the {len(fs)} tokens that are ever a favourite")
ax.set_xlabel("seat"); ax.set_ylabel("key")
ax.set_title("A head of width one can only pick the largest or the smallest key so far", fontsize=11.5)
ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
save(fig, os.path.join(OUT, "fig09-14-width-one.png"))

# 15 Lab 9 Breakage 2: what the naive removal takes away
fig, ax = plt.subplots(figsize=(8, 4.2))
vals = [9.508, 8.952, 5.231]
labs = ["head 0's whole\ncontribution\n(correct removal)", "taken away by\nzeroing 32 columns\nafter W_o", "of which really\nhead 0's"]
cols = [BLUE, CORAL, PLUM]
ax.barh(range(3)[::-1], vals, color=[TINT[c] if c in TINT else LIGHT for c in cols], edgecolor=cols, hatch="", lw=1.8)
for i, v in enumerate(vals):
    ax.text(v + 0.15, 2 - i, f"{v:.3f}", va="center", fontsize=10, color=NAVY)
ax.set_yticks(range(3)[::-1], labs, fontsize=9.5); ax.set_xlim(0, 11.5); ax.set_xlabel("size of the change to the layer's output")
ax.set_title("Removing a head after the output table removes the wrong thing", fontsize=11.5)
save(fig, os.path.join(OUT, "fig09-15-wrong-place.png"))
print("ch09 figures:", len(os.listdir(OUT)))
