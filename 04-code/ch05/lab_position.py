"""Lab 5: scramble the sentence, watch the rows fail to notice, then add positions.
Run from 04-code/ch05:  python3 lab_position.py   (writes outputs/lab5_*.png)"""
import _common
from _common import SENTENCE, SCRAMBLED, TINY_BPE, HERE
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tiny.tokenizer import TOY_VOCAB, load_bpe, TINY_VOCAB_SIZE
from tiny.embed import (TOY_EMBED, TOY_POS, dot, add_rows, add_positions,
                        Embedding, PositionTable, sinusoidal, embed_with_positions)

def rows(text):
    return TOY_EMBED[[TOY_VOCAB.index(w) for w in text.split()]]

def agreements(table):
    """Every token's dot product with every token: the raw material of attention (Chapter 8).
    Returned as a sorted list, because an order-blind reader sees only *which* numbers occur."""
    return sorted(round(float(dot(a, b)), 3) for a in table for b in table)

def same(a, b):
    return agreements(a) == agreements(b)

def plot(table, words, path, colour):
    fig, ax = plt.subplots(figsize=(4, 4))
    for w, (x, y) in zip(words, table):
        ax.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="->", color=colour, lw=2))
        ax.text(x * 1.08, y * 1.08 + 0.08, w, ha="center")
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.axhline(0, c="grey", lw=.5); ax.axvline(0, c="grey", lw=.5)
    ax.set_aspect("equal"); fig.savefig(path, dpi=100); plt.close(fig)

os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
A, B = rows(SENTENCE), rows(SCRAMBLED)
print("Step 1, no positions:")
print("  same bag of rows:", sorted(A.tolist()) == sorted(B.tolist()))
print("  same 36 agreements:", same(A, B))
pA, pB = add_positions(A, TOY_POS), add_positions(B, TOY_POS)
print("Step 2, toy positions added:")
print("  same bag of rows:", sorted(pA.tolist()) == sorted(pB.tolist()))
print("  same 36 agreements:", same(pA, pB))
diff = sum(1 for x, y in zip(agreements(pA), agreements(pB)) if x != y)
print(f"  agreements that differ: {diff} of 36")
words = [f"{w}@{k}" for k, w in enumerate(SENTENCE.split())]
plot(pA, words, os.path.join(HERE, "outputs", "lab5_positions.png"), "#1f3f66")
print("Step 3: drew outputs/lab5_positions.png;", len({tuple(r) for r in pA.tolist()}), "different arrows")

print("\nBREAKAGE 1: every position gets the same row, (0.5, 0.5)")
flat = np.tile([0.5, 0.5], (6, 1))
fA, fB = add_positions(A, flat), add_positions(B, flat)
print("  the two 'the's:", fA[0].tolist(), fA[4].tolist())
print("  same 36 agreements as the scrambled sentence:", same(fA, fB))
plot(fA, words, os.path.join(HERE, "outputs", "lab5_broken.png"), "#e4573d")

print("BREAKAGE 2: add the positions after summing the rows")
def total(table):
    return [float(sum(r[i] for r in table)) for i in range(2)]
pos_total = total(TOY_POS)
print("  sum of rows + sum of positions:", add_rows(total(A), pos_total), "vs", add_rows(total(B), pos_total))
print("  and summing after adding:      ", total(pA), "vs", total(pB))

tok = load_bpe(TINY_BPE)
ids = tok.encode(SENTENCE)
emb = Embedding(TINY_VOCAB_SIZE, seed=1865)
pos = PositionTable(seed=1865)
X = embed_with_positions(ids, emb, pos)
S = sinusoidal(len(ids))
print("\nCHECKPOINT")
print("toy rows + positions:", pA.tolist())
print(f"Tiny: positions {pos.shape}, row 0 starts {np.round(pos.table[0][:3], 4).tolist()}, "
      f"block {X.shape}, block sum {X.sum():.4f}")
print("sinusoidal position 4 starts", np.round(S[4][:4], 4).tolist())
