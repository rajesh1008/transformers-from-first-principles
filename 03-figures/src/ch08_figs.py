"""Figures for Chapter 8 (Stage 0 voice reference so far): the 6x6 score table.
Run from the project root: python3 03-figures/src/ch08_figs.py"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
setup()
OUT = os.path.join(ROOT, "03-figures", "ch08"); os.makedirs(OUT, exist_ok=True)
words = ["the", "cat", "sat", "on", "the", "mat"]
X = np.array([[1, 0], [0, 1], [1, 1], [1, -1], [1, 0], [-1, 1]])
W_q = np.array([[1, 0], [0, 2]]); W_k = np.array([[0, 1], [1, 0]])
S = (X @ W_q) @ (X @ W_k).T
fig, ax = plt.subplots(figsize=(6.4, 5.6))
cmap = LinearSegmentedColormap.from_list("plum", ["#fbe6de", "#ffffff", PLUM])
im = ax.imshow(S, cmap=cmap, vmin=-3, vmax=3)
for i in range(6):
    for j in range(6):
        ax.text(j, i, str(S[i, j]), ha="center", va="center", fontsize=12, color=NAVY, weight="bold" if (i, j) == (2, 1) else "normal")
ax.add_patch(plt.Rectangle((0.5, 1.5), 1, 1, fill=False, ec=CORAL, lw=2.4))
ax.set_xticks(range(6), [f"{w}\n{k}" for k, w in enumerate(words)]); ax.set_yticks(range(6), [f"{k} {w}" for k, w in enumerate(words)])
ax.set_xlabel("the word being looked at (its key)"); ax.set_ylabel("the word doing the looking (its query)")
ax.grid(False); ax.set_title("Thirty-six agreements: the score table for one head")
save(fig, os.path.join(OUT, "fig08-01-score-table.png"))
print("ch08 figures:", len(os.listdir(OUT)))

# ---------------------------------------------------------------- full chapter figures (10 Nov 2026)
sys.path.insert(0, os.path.join(ROOT, "04-code"))
from tiny.attention import softmax, causal_mask, attention
from tiny.embed import TOY_POS
W_v = np.array([[1, 0], [1, 1]])
Q, K, V = X @ W_q, X @ W_k, X @ W_v
Ss = S / np.sqrt(2)
A_open = softmax(Ss); M = causal_mask(6); A = softmax(np.where(M, Ss, -np.inf))
labels_x = [f"{w}\n{k}" for k, w in enumerate(words)]; labels_y = [f"{k} {w}" for k, w in enumerate(words)]
plum_map = LinearSegmentedColormap.from_list("sh", ["#ffffff", "#efe7f5", PLUM])

def heat(M_, title_, path, fmt="{:.3f}", mask=None, hl=None, cmap=plum_map, vmin=0, vmax=1, rows=labels_y, cols=labels_x):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.imshow(M_, cmap=cmap, vmin=vmin, vmax=vmax)
    n = M_.shape[0]
    for i in range(n):
        for j in range(n):
            if mask is not None and not mask[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc="#f3f4f6", ec="#c7ccd4", hatch="///", lw=0.4))
                ax.text(j, i, "masked", ha="center", va="center", fontsize=7, color=GREY)
            else:
                ax.text(j, i, fmt.format(M_[i, j]), ha="center", va="center", fontsize=10.5, color=NAVY)
    if hl:
        ax.add_patch(plt.Rectangle((hl[1] - .5, hl[0] - .5), 1, 1, fill=False, ec=CORAL, lw=2.4))
    ax.set_xticks(range(n), cols); ax.set_yticks(range(n), rows)
    ax.set_xlabel("the word being looked at (its key)"); ax.set_ylabel("the word doing the looking (its query)")
    ax.grid(False); ax.set_title(title_)
    save(fig, os.path.join(OUT, path))

# 02 queries and keys as arrows
fig, axs = plt.subplots(1, 2, figsize=(9, 4.6))
for ax, R, name, col in [(axs[0], Q, "queries: the question each word asks", BLUE), (axs[1], K, "keys: the label each word wears", TEAL)]:
    for (a, b), w in zip(R, words):
        ax.annotate("", (a, b), (0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
        ax.text(a * 1.08 + 0.05, b * 1.08 + 0.05, w, color=NAVY, fontsize=10)
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.4, 2.4); ax.set_aspect("equal"); ax.axhline(0, color="#c7ccd4", lw=1); ax.axvline(0, color="#c7ccd4", lw=1)
    ax.set_title(name, fontsize=11.5)
save(fig, os.path.join(OUT, "fig08-02-queries-keys.png"))

# 03 sat's row: scores, exponentials, shares
row = Ss[2]; e = np.exp(row); sh = e / e.sum()
fig, axs = plt.subplots(1, 3, figsize=(10, 3.6), sharey=False)
for ax, vals, t, col, hatch in [(axs[0], row, "scaled scores", NAVY, ""), (axs[1], e, "exponentials", AMBER, ".."), (axs[2], sh, "shares (sum to one)", PLUM, "//")]:
    ax.bar(range(6), vals, color=TINT.get(col, LIGHT), edgecolor=col, hatch=hatch, lw=1.6)
    for k, v in enumerate(vals):
        ax.text(k, v + (0.05 * max(vals) if v >= 0 else -0.12 * max(abs(vals))), f"{v:.2f}" if t != "shares (sum to one)" else f"{v:.3f}", ha="center", fontsize=8.5, color=NAVY)
    ax.set_xticks(range(6), words, fontsize=9); ax.set_title(t, fontsize=11.5)
save(fig, os.path.join(OUT, "fig08-03-sat-softmax.png"))

heat(A_open, "Shares with no mask: every row sums to one", "fig08-04-shares-open.png", hl=(2, 1))
heat(M.astype(float), "The causal mask: each token may look at itself and earlier tokens", "fig08-05-mask.png", fmt="{:.0f}", cmap=LinearSegmentedColormap.from_list("m", ["#ffffff", "#dff2ee"]))
heat(A, "The Attention Table: masked, scaled shares for one head", "fig08-06-attention-table.png", mask=M, hl=(2, 1))

# 07 sat's output as a weighted sum of value arrows
fig, ax = plt.subplots(figsize=(6, 5))
start = np.zeros(2)
for j, col in zip(range(3), [BLUE, TEAL, AMBER]):
    piece = A[2, j] * V[j]
    ax.annotate("", start + piece, start, arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4))
    ax.text(*(start + piece / 2 + np.array([0.02, 0.06])), f"{A[2, j]:.3f} x {words[j]} {tuple(int(v) for v in V[j])}", fontsize=9, color=NAVY)
    start = start + piece
ax.annotate("", start, (0, 0), arrowprops=dict(arrowstyle="-|>", color=PLUM, lw=2.8, ls="--"))
ax.text(start[0] - 0.55, start[1] + 0.08, f"sat's output {tuple(round(float(v), 3) for v in start)}", color=PLUM, fontsize=10, weight="bold")
ax.annotate("", (1, 1), (0, 0), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.6, ls=":"))
ax.text(0.85, 1.05, "sat's own row (1, 1)", color=GREY, fontsize=9)
ax.set_xlim(-0.1, 1.9); ax.set_ylim(-0.1, 1.2); ax.set_aspect("equal"); ax.set_title("sat's output: three value arrows, each shrunk by its share")
save(fig, os.path.join(OUT, "fig08-07-weighted-sum.png"))

# 08 saturation (numbers from 04-code/ch08/outputs/saturation.txt)
widths = [2, 16, 64, 128, 512, 1024]
top_raw = [0.455, 0.773, 0.884, 0.925, 0.958, 0.975]; top_sc = [0.380, 0.413, 0.416, 0.423, 0.420, 0.425]
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.plot(widths, top_raw, color=CORAL, marker="o", ls="--", label="not divided by the square root of the width")
ax.plot(widths, top_sc, color=NAVY, marker="s", label="divided (scaled)")
ax.set_xscale("log", base=2); ax.set_ylim(0, 1.05); ax.set_xlabel("key width d (numbers per query and key)"); ax.set_ylabel("average largest share in a row of six")
ax.text(1024, 0.99, "saturated", color=CORAL, ha="right"); ax.text(1024, 0.45, "stays spread", color=NAVY, ha="right")
ax.axhline(1 / 6, color=GREY, ls=":", lw=1); ax.text(2.1, 1 / 6 + 0.02, "even split, 1/6", color=GREY, fontsize=9)
ax.set_title("Breakage: remove the scaling and the shares saturate")
save(fig, os.path.join(OUT, "fig08-08-saturation.png"))

# 09 cheating
fig, ax = plt.subplots(figsize=(6.4, 4))
vals = [100.0, 4.8, 1.8]; labs = ["look-ahead head,\nmask off", "same head,\nmask on", "next token equals\nthis token"]
cols = [CORAL, NAVY, GREY]; hatches = ["xx", "", ".."]
for k, (v, c, h) in enumerate(zip(vals, cols, hatches)):
    ax.bar(k, v, color=TINT.get(c, LIGHT), edgecolor=c, hatch=h, lw=1.8); ax.text(k, v + 2, f"{v:.1f}%", ha="center", color=NAVY, weight="bold")
ax.set_xticks(range(3), labs); ax.set_ylabel("next tokens of the novel guessed right (%)"); ax.set_ylim(0, 112)
ax.set_title("Breakage: without the mask a head can read the answer")
save(fig, os.path.join(OUT, "fig08-09-cheating.png"))

# 10 GPT-2 heads, from 04-code/ch08/gpt2_head.py's printed tables
pieces = ["the", "␣cat", "␣sat", "␣on", "␣the", "␣mat"]
g00 = np.array([[1, 0, 0, 0, 0, 0], [.731, .269, 0, 0, 0, 0], [.477, .139, .384, 0, 0, 0], [.397, .202, .358, .043, 0, 0], [.392, .201, .319, .039, .049, 0], [.282, .103, .046, .110, .138, .321]])
g411 = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, .999, .001, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]])
fig, axs = plt.subplots(1, 2, figsize=(10.5, 5))
for ax, Mg, t in [(axs[0], g00, "GPT-2 layer 0, head 0"), (axs[1], g411, "GPT-2 layer 4, head 11")]:
    ax.imshow(Mg, cmap=plum_map, vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            if j > i:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc="#f3f4f6", ec="#c7ccd4", hatch="///", lw=0.4))
            else:
                ax.text(j, i, f"{Mg[i, j]:.2f}", ha="center", va="center", fontsize=9, color=NAVY)
    ax.set_xticks(range(6), pieces, fontsize=9); ax.set_yticks(range(6), pieces, fontsize=9); ax.grid(False); ax.set_title(t, fontsize=12)
save(fig, os.path.join(OUT, "fig08-10-gpt2-heads.png"))

# 11 first-token share by layer
first = [0.35, 0.58, 0.58, 0.69, 0.70, 0.79, 0.80, 0.85, 0.84, 0.87, 0.88, 0.75]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(12), first, color=TINT[SLATEBLUE], edgecolor=SLATEBLUE, lw=1.6)
ax.axhline(0.29, color=GREY, ls=":", lw=1.4); ax.text(11.4, 0.31, "even split: 0.29", color=GREY, ha="right", fontsize=9)
ax.set_xticks(range(12)); ax.set_xlabel("GPT-2 layer"); ax.set_ylabel("average share on the first token")
ax.set_title("Where unwanted share goes: the first token of the running sentence")
save(fig, os.path.join(OUT, "fig08-11-first-token.png"))

# 12 Signature 3: the head as a pipeline
fig, ax = canvas(10, 4.2)
box(ax, 2, 16, 12, 10, "X", NAVY, sub="(6, 2) rows")
for k, (nm, sub, col) in enumerate([("Q = X W_q", "asks", BLUE), ("K = X W_k", "labels", TEAL), ("V = X W_v", "offers", AMBER)]):
    box(ax, 20, 30 - k * 12, 16, 9, nm, col, sub=sub, fs=10)
    arrow(ax, 14, 21, 20, 34.5 - k * 12)
box(ax, 42, 24, 14, 10, "Q K.T / √d", NAVY, sub="scores (6, 6)", fs=10)
arrow(ax, 36, 34.5, 42, 30); arrow(ax, 36, 22.5, 42, 27)
box(ax, 60, 24, 10, 10, "mask", CORAL, sub="no peeking", fs=10)
arrow(ax, 56, 29, 60, 29)
box(ax, 74, 24, 11, 10, "softmax", PLUM, sub="shares", fs=10)
arrow(ax, 70, 29, 74, 29)
box(ax, 88, 12, 10, 10, "@ V", PLUM, sub="(6, 2) out", fs=10)
arrow(ax, 79.5, 24, 91, 22); arrow(ax, 36, 10.5, 88, 15)
label(ax, 50, 4, "Signature 3: one head. Every token's output is a share-weighted average of the values it may see.", fs=10, color=NAVY)
save(fig, os.path.join(OUT, "fig08-12-signature3.png"))

# 13 Lab 8 checkpoint table (seed 1865)
rng = np.random.default_rng(1865)
Wq2, Wk2, Wv2 = (np.round(rng.normal(size=(2, 2)), 2) for _ in range(3))
_, AL = attention(X + TOY_POS, Wq2, Wk2, Wv2)
heat(AL, "Lab 8 checkpoint: the Attention Table for the seeded tables", "fig08-13-lab-checkpoint.png", mask=M)
print("ch08 figures:", len(os.listdir(OUT)))
