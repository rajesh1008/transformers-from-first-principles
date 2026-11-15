"""Lab 4: plot the six embeddings of the running sentence, then break them all into one arrow.
Run from 04-code/ch04:  python3 lab_embed.py   (writes outputs/lab4_six_arrows.png)"""
import _common
from _common import SENTENCE, TINY_BPE, HERE
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tiny.tokenizer import TOY_VOCAB, load_bpe, TINY_VOCAB_SIZE
from tiny.embed import TOY_EMBED, Embedding, cosine, dot

ids = [TOY_VOCAB.index(w) for w in SENTENCE.split()]
X = TOY_EMBED[ids]

def show(table, label):
    print(f"{label}:")
    for w, row in zip(SENTENCE.split(), table):
        print(f"  {w:<4} ({row[0]:>2.0f}, {row[1]:>2.0f})")

def plot(table, path, colour):
    fig, ax = plt.subplots(figsize=(4, 4))
    for w, (x, y) in zip(SENTENCE.split(), table):
        ax.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="->", color=colour, lw=2))
        ax.text(x * 1.1, y * 1.1 + 0.08, w, ha="center")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.axhline(0, c="grey", lw=.5); ax.axvline(0, c="grey", lw=.5)
    ax.set_aspect("equal"); fig.savefig(path, dpi=100); plt.close(fig)

def agreements(table):
    cat = table[1]
    return [float(round(cosine(cat, row), 3)) for row in table]

show(X, "Step 1: six ids, six rows")
os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
plot(X, os.path.join(HERE, "outputs", "lab4_six_arrows.png"), "#1f3f66")
print("Step 2: drew outputs/lab4_six_arrows.png;", len(set(map(tuple, X))), "different arrows")
print("Step 3: cosine of cat with each:", agreements(X))
flipped = [TOY_VOCAB.index(w) for w in "the mat sat on the cat".split()]
print("Step 4: 'the mat sat on the cat' uses the same rows in a new order:",
      sorted(map(tuple, TOY_EMBED[flipped].astype(int).tolist())) == sorted(map(tuple, X.astype(int).tolist())))

# The breakage: every word gets the same row
broken = np.tile([1.0, 1.0], (len(TOY_VOCAB), 1))
B = broken[ids]
print("\nBREAKAGE: every row of the table set to (1, 1)")
show(B, "the six rows now")
plot(B, os.path.join(HERE, "outputs", "lab4_broken.png"), "#e4573d")
print("different arrows:", len(set(map(tuple, B))))
print("cosine of cat with each:", agreements(B))
h = (-1.2, 1.0)
print("scores for every next token from h =", h, ":", sorted({float(round(dot(h, r), 1)) for r in broken}),
      "(one value shared by all 12)")
print("'the cat sat' and 'the dog sat' look identical:",
      np.array_equal(broken[[0, 1, 2]], broken[[0, 6, 2]]))

tok = load_bpe(TINY_BPE)
emb = Embedding(TINY_VOCAB_SIZE, seed=1865)
T = emb(tok.encode(SENTENCE))
print("\nCHECKPOINT")
print("toy 6x2 table:", X.astype(int).tolist())
print(f"Tiny: table {emb.shape}, sentence block {T.shape}, row 725 starts {np.round(emb.table[725][:3], 4).tolist()}, "
      f"block sum {T.sum():.4f}")
