"""Lab 6: rotary positions on the running sentence; show the agreements depend only on the gap; then break it.
Run from 04-code/ch06:  python3 lab_rotary.py   (writes outputs/lab6_*.png)"""
import _common
from _common import SENTENCE, SCRAMBLED, TINY_BPE, HERE
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tiny.tokenizer import TOY_VOCAB, load_bpe, TINY_VOCAB_SIZE
from tiny.embed import TOY_EMBED, TOY_TURN, apply_rotary, agreement_table, Embedding, Rotary, length

def rows(text):
    return TOY_EMBED[[TOY_VOCAB.index(w) for w in text.split()]]

def table(X, start=0, asking_turned=True, answering_turned=True, same_angle=False):
    where = [0] * len(X) if same_angle else list(range(start, start + len(X)))
    turned = apply_rotary(X, where, speeds=TOY_TURN)
    return agreement_table(turned if asking_turned else X, turned if answering_turned else X)

def show(t, words):
    print("        " + "".join(f"{w:>7}" for w in words))
    for w, r in zip(words, t):
        print(f"  {w:>5} " + "".join(f"{x:7.3f}" for x in r))

def heat(t, words, path, cmap):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(t, cmap=cmap, vmin=-2, vmax=2)
    ax.set_xticks(range(6), words); ax.set_yticks(range(6), words)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{t[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)

os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
words = SENTENCE.split()
A, B = rows(SENTENCE), rows(SCRAMBLED)
T0 = table(A, start=0)
print("Step 1: the 6 x 6 agreement table, every row turned 30 degrees per position (row asks, column answers)")
show(T0, words)
T100 = table(A, start=100)
print("Step 2: the same sentence at positions 100-105; largest change in any agreement:",
      f"{np.abs(T0 - T100).max():.1e}")
TB = table(B, start=0)
print("Step 3: the scrambled sentence; agreements that differ from the running sentence:",
      int((np.abs(np.sort(T0, axis=None) - np.sort(TB, axis=None)) > 1e-9).sum()), "of 36")
heat(T0, words, os.path.join(HERE, "outputs", "lab6_table.png"), "PuOr")

print("\nBREAKAGE 1: turn the asking rows (queries) but not the answering rows (keys)")
H0, H100 = table(A, 0, answering_turned=False), table(A, 100, answering_turned=False)
print("  agreements that change when the sentence moves to positions 100-105:",
      int((np.abs(H0 - H100) > 1e-9).sum()), "of 36")
print("  the -> cat (gap 1) at positions 0,1:", round(H0[0, 1], 3), " at 100,101:", round(H100[0, 1], 3))
heat(H100 - H0, words, os.path.join(HERE, "outputs", "lab6_broken.png"), "RdBu")
print("BREAKAGE 2: the same angle for every position")
S0, SB = table(A, same_angle=True), table(B, same_angle=True)
print("  agreements that differ from the scrambled sentence:",
      int((np.abs(np.sort(S0, axis=None) - np.sort(SB, axis=None)) > 1e-9).sum()), "of 36")

tok = load_bpe(TINY_BPE)
ids = tok.encode(SENTENCE)
emb = Embedding(TINY_VOCAB_SIZE, seed=1865)
X = emb(ids)
R = Rotary()(X)
print("\nCHECKPOINT")
print("toy table, row 'the'@0:", np.round(T0[0], 3).tolist())
print("toy table, row 'mat'@5:", np.round(T0[5], 3).tolist())
print(f"Tiny: rotary on block {R.shape}, row 1 starts {np.round(R[1][:3], 4).tolist()}, block sum {R.sum():.4f}, "
      f"lengths kept: {bool(np.allclose([length(r) for r in R], [length(x) for x in X]))}, learned numbers 0")
