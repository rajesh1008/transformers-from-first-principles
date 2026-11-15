"""Lab 7: the 6 x 6 agreement table by hand (loops), then in one line; break it by mismatching shapes;
the checkpoint on Tiny's 7 x 128 block (token table seed 1865).   Run from 04-code/ch07:  python3 lab_matrices.py"""
import _common
from _common import SENTENCE, TINY_BPE
import numpy as np
from tiny.tokenizer import TOY_VOCAB, load_bpe, TINY_VOCAB_SIZE
from tiny.embed import TOY_EMBED, TOY_TURN, Embedding, apply_rotary, agreement_table
from tiny.linalg import agreement_table_fast

SEED = 1865
words = SENTENCE.split()
X = TOY_EMBED[[TOY_VOCAB.index(w) for w in words]]               # (6, 2): six tokens, two numbers each

print("STEP 1: by hand (Chapter 6's loops, 36 dot products)")
slow = agreement_table(X)
for w, r in zip(words, slow):
    print(f"  {w:>4} " + " ".join(f"{v:4g}" for v in r))

print("STEP 2: in one line, X @ X.T")
fast = X @ X.T
print("  shapes:", X.shape, "@", X.T.shape, "->", fast.shape, "| identical to step 1:", bool(np.array_equal(slow, fast)))

print("STEP 3: Chapter 6's rotary rows, the same line")
Xr = apply_rotary(X, range(6), speeds=TOY_TURN)
print("  row the@0:", np.round(Xr @ Xr.T, 3)[0].tolist())

print("\nBREAKAGE: mismatch the shapes")
tok = load_bpe(TINY_BPE)
ids = tok.encode(SENTENCE)
emb = Embedding(TINY_VOCAB_SIZE, seed=SEED)
B = emb(ids)                                                       # (7, 128)
for label, f in [("X @ X", lambda: X @ X), ("B @ emb.table  (score Tiny's 7 rows against its 2,048 rows)",
                                            lambda: B @ emb.table)]:
    try:
        f()
    except ValueError as e:
        print(f"  {label}: ValueError: {e}")
wrong = X.T @ X                                                    # a 'fix' that runs
print("  X.T @ X runs, but its shape is", wrong.shape, "not (6, 6):", wrong.tolist())
try:
    assert fast.shape == (len(words), len(words)), "table must be tokens x tokens"
    assert wrong.shape == (len(words), len(words)), "table must be tokens x tokens"
except AssertionError as e:
    print("  a one-line shape check catches it: AssertionError:", e)

print("  the fix for Tiny's scores turns the table:", B.shape, "@", emb.table.T.shape, "->", (B @ emb.table.T).shape)

print(f"\nCHECKPOINT (seed {SEED})")
TB = agreement_table_fast(B)
print("  toy table row the:", fast[0].tolist(), "| row mat:", fast[5].tolist())
print("  Tiny ids:", ids, "block", B.shape, "-> table", TB.shape)
print("  first row:", "[" + ", ".join(f"{v:.4f}" for v in TB[0]) + "]")
print(f"  table sum: {TB.sum():.4f} | diagonal sum: {np.trace(TB):.4f} | mirror image: {bool(np.allclose(TB, TB.T))}")
