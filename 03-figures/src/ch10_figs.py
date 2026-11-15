"""Figures for Chapter 10, Attention Variants.  Every number is computed here from the chapter's own code
(04-code/ch10, seed 1865) or read from the outputs it saved.
Run from the project root: python3 03-figures/src/ch10_figs.py"""
import os, sys, re, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
setup()
OUT = os.path.join(ROOT, "03-figures", "ch10"); os.makedirs(OUT, exist_ok=True)
CODE = os.path.join(ROOT, "04-code"); CH = os.path.join(CODE, "ch10")
sys.path.insert(0, CODE); sys.path.insert(0, CH)
from tiny.attention import softmax, causal_mask, window_mask, MultiHead
words = ["the", "cat", "sat", "on", "the", "mat"]
X = np.array([[1, 0], [0, 1], [1, 1], [1, -1], [1, 0], [-1, 1]], dtype=float)
W_q = np.array([[1, 0], [0, 2]], float); W_k = np.array([[0, 1], [1, 0]], float); W_v = np.array([[1, 0], [1, 1]], float)
plum_map = LinearSegmentedColormap.from_list("sh", ["#ffffff", "#efe7f5", PLUM])
S = (X @ W_q) @ (X @ W_k).T / np.sqrt(2)

def heat(ax, A, M, t, fs=9.5):
    n = A.shape[0]; ax.imshow(np.where(M, A, 0), cmap=plum_map, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if not M[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc="#f3f4f6", ec="#c7ccd4", hatch="///", lw=0.4))
            elif fs:
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=fs, color=NAVY)
    ax.set_xticks(range(n), words[:n], fontsize=9); ax.set_yticks(range(n), words[:n], fontsize=9)
    ax.grid(False); ax.set_title(t, fontsize=11)

# 01 cache per token and at length, GPT-2 small's shape
fig, ax = plt.subplots(figsize=(8.4, 4.6))
lengths = np.array([1024, 4096, 16384, 65536])
for kv, name, col in [(12, "multi-head (12 kv heads)", VERM), (4, "grouped-query (4)", BLUE), (1, "multi-query (1)", TEAL)]:
    mib = 12 * kv * 64 * 2 * 2 * lengths / 2**20
    ax.plot(lengths, mib, marker="o", color=col, label=name)
    ax.text(lengths[-1] * 1.1, mib[-1], f"{mib[-1]:,.0f} MiB", va="center", fontsize=9.5, color=col)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(lengths, [f"{l:,}" for l in lengths])
ax.set_xlabel("tokens held"); ax.set_ylabel("key-value cache, MiB (16-bit)")
ax.set_title("GPT-2 small's shape: the cache grows with every token kept"); ax.legend(loc="upper left")
ax.set_xlim(800, 2**18)
save(fig, os.path.join(OUT, "fig10-01-cache-growth.png"))

# 02 sharing diagram
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6))
for ax, (t, kv) in zip(axs, [("multi-head", [0, 1, 2, 3]), ("grouped-query, 2 groups", [0, 0, 1, 1]), ("multi-query", [0, 0, 0, 0])]):
    ax.set_xlim(0, 40); ax.set_ylim(0, 30); ax.axis("off"); ax.set_title(t, fontsize=11.5)
    for h in range(4):
        box(ax, 2 + h * 9.5, 20, 7.5, 6, f"q{h}", BLUE, fs=10)
    used = sorted(set(kv))
    for g in used:
        x = 2 + (g * 4 / max(len(used), 1)) * 9.5 if len(used) < 4 else 2 + g * 9.5
        xs = {4: 2 + g * 9.5, 2: 6.75 + g * 19, 1: 16.25}[len(used)]
        box(ax, xs, 3, 7.5, 6, f"k,v{g}", SLATEBLUE, fs=10)
        for h in range(4):
            if kv[h] == g:
                arrow(ax, xs + 3.75, 9, 2 + h * 9.5 + 3.75, 20, color=SLATE, lw=1.2, ms=10)
    ax.text(20, 0.5, f"cached per token per layer: {len(used) * 2} blocks", ha="center", fontsize=9.5, color=PLUM)
save(fig, os.path.join(OUT, "fig10-02-sharing.png"))

# 03 three masks on the running sentence
def gmask(n, w, g):
    M = window_mask(n, w).copy(); C = causal_mask(n)
    for s in g:
        M[:, s] |= C[:, s]; M[s, :] |= C[s, :]
    return M
fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, (t, M) in zip(axs, [("causal, no window", causal_mask(6)), ("window 3", window_mask(6, 3)), ("window 3 + global seat 1 (cat)", gmask(6, 3, [1]))]):
    A = softmax(np.where(M, S, -np.inf)); heat(ax, A, M, t)
    ax.add_patch(plt.Rectangle((0.5, 4.5), 1, 1, fill=False, ec=CORAL, lw=2.2))
save(fig, os.path.join(OUT, "fig10-03-masks.png"))

# 04 reach across layers
fig, ax = plt.subplots(figsize=(8.4, 3.8))
n, w = 24, 4
for L in range(1, 5):
    reach = min(n - 1, L * (w - 1))
    ax.barh(L, reach + 1, left=n - 1 - reach, color=TINT[BLUE], edgecolor=BLUE)
    ax.text(n - 1 - reach - 0.4, L, f"{reach} back", ha="right", va="center", fontsize=9.5, color=NAVY)
ax.set_yticks(range(1, 5), [f"after layer {L}" for L in range(1, 5)]); ax.set_xlim(-1, n)
ax.set_xlabel("seat (the last token sits at 23)"); ax.invert_yaxis()
ax.set_title("Window 4: each layer reaches three more seats back")
save(fig, os.path.join(OUT, "fig10-04-reach.png"))

# 05 interleaving cost
fig, ax = plt.subplots(figsize=(8.4, 3.8))
rows = [("all 12 full", 2304.0), ("1 in 4 full", 582.8), ("1 in 6 full", 391.5), ("all 12 window 256", 9.0)]
txt = open(os.path.join(CH, "outputs", "local_global.txt")).read()
vals = [float(x) for x in re.findall(r"\s([\d.]+) MiB", txt)]
assert len(vals) == 4
ax.barh(range(4), vals, color=[TINT[VERM], TINT[BLUE], TINT[BLUE], TINT[TEAL]], edgecolor=[VERM, BLUE, BLUE, TEAL])
for i, v in enumerate(vals):
    ax.text(v + 20, i, f"{v:,.1f} MiB", va="center", fontsize=9.5, color=NAVY)
ax.set_yticks(range(4), [r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0, 2800)
ax.set_xlabel("cache at 65,536 tokens, GPT-2 small's shape, 16-bit")
ax.set_title("The full layers pay almost all the memory")
save(fig, os.path.join(OUT, "fig10-05-interleave.png"))

# 06 recall: softmax over a cache vs a fixed linear state
txt = open(os.path.join(CH, "outputs", "linear.txt")).read()
rec = re.findall(r"n=\s*(\d+): softmax over cache\s+([\d.]+) \| linear state\s+([\d.]+)", txt)
ns = [int(a) for a, _, _ in rec]; so = [float(b) for _, b, _ in rec]; li = [float(c) for _, _, c in rec]
fig, ax = plt.subplots(figsize=(8.4, 4.2))
ax.plot(ns, so, marker="o", color=SLATEBLUE, label="softmax over the cache")
ax.plot(ns, li, marker="s", color=VERM, label="fixed linear state (16 x 16)")
ax.plot(ns, [1 / k for k in ns], ls=":", color=GREY, label="chance")
ax.axvline(16, color=GREY, lw=1); ax.text(17, 0.5, "pairs = state width", fontsize=9.5, color=GREY)
ax.set_xscale("log", base=2); ax.set_xticks(ns, ns); ax.set_ylim(0, 1.08)
ax.set_xlabel("key-value pairs stored"); ax.set_ylabel("share recalled correctly")
ax.set_title("A state that never grows forgets once it is full"); ax.legend(loc="lower left")
save(fig, os.path.join(OUT, "fig10-06-recall.png"))

# 07 tree toy
txt = open(os.path.join(CH, "outputs", "sparse_tree.txt")).read()
tr = re.findall(r"\s+(\d+)\s+\|\s+([\d,]+)\s+\|\s+([\d.]+)%\s+\|\s+([\d.]+)\s+needles opened (\d)/8", txt)
read = [int(b.replace(",", "")) for _, b, _, _, _ in tr]; err = [float(d) for *_, d, _ in tr]; nd = [int(e) for *_, e in tr]
fig, ax = plt.subplots(figsize=(8.4, 4.2))
ax.plot(read, err, marker="o", color=PLUM, label="error against full attention")
ax2 = ax.twinx(); ax2.bar(read, nd, width=[r * 0.12 for r in read], color=TINT[AMBER], edgecolor=AMBER, alpha=0.8, label="needles opened (of 8)")
ax2.set_ylim(0, 9); ax2.set_ylabel("needles opened"); ax2.grid(False)
ax.set_xscale("log"); ax.set_xlabel("entries read (of 4,096 keys)"); ax.set_ylabel("relative error")
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
ax.set_title("Mean-pooled summaries find few needles until most of the cache is read")
save(fig, os.path.join(OUT, "fig10-07-tree.png"))

# 08 lab checkpoint: Tiny's four heads, window 3, on the running sentence
sys.path.insert(0, CH)
from tiny.embed import Embedding, PositionTable
from tiny.tokenizer import load_bpe
tok = load_bpe(os.path.join(CODE, "tiny", "tiny_bpe_2048.txt")); ids = tok.encode("the cat sat on the mat")
Xt = Embedding(len(tok))(ids) * 50 + PositionTable()(range(len(ids))) * 50
_, A = MultiHead(n_heads=4, n_kv_heads=2, window=3)(Xt, return_shares=True)
n = len(ids); names = [tok.decode([i]).strip() or "_" for i in ids]
fig, axs = plt.subplots(1, 4, figsize=(14, 3.9))
for h, ax in enumerate(axs):
    M = window_mask(n, 3); ax.imshow(np.where(M, A[h], 0), cmap=plum_map, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if not M[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc="#f3f4f6", ec="#c7ccd4", hatch="///", lw=0.4))
            else:
                ax.text(j, i, f"{A[h, i, j]:.2f}", ha="center", va="center", fontsize=7, color=NAVY)
    ax.set_xticks(range(n), names, fontsize=7, rotation=90); ax.set_yticks(range(n), names, fontsize=7); ax.grid(False)
    ax.set_title(f"head {h} (kv block {h // 2})", fontsize=10.5)
save(fig, os.path.join(OUT, "fig10-08-lab-checkpoint.png"))
print("ch10 figures written to", OUT, "| tokens:", n)
