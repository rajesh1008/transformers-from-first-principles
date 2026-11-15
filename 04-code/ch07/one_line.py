"""Section 7.1 / Interlude: the whole agreement table in one line, checked against Chapter 6's loops;
the rotary table of Chapter 6 the same way; Tiny's 7 x 7 table.   Run: python3 one_line.py"""
import _common
from _common import SENTENCE, TINY_BPE
import numpy as np
from tiny.tokenizer import TOY_VOCAB, load_bpe, TINY_VOCAB_SIZE
from tiny.embed import TOY_EMBED, TOY_TURN, apply_rotary, agreement_table, Embedding

X = TOY_EMBED[[TOY_VOCAB.index(w) for w in SENTENCE.split()]]
T = X @ X.T                                           # all 36 agreements in one line
print("X @ X.T, row the:", T[0].tolist())
print("same as Chapter 6's loops:", bool(np.array_equal(T, agreement_table(X))))

Xr = apply_rotary(X, range(6), speeds=TOY_TURN)       # Chapter 6: turn each row 30 degrees per seat
Tr = Xr @ Xr.T
print("rotary rows, X_rot @ X_rot.T, row the@0:", np.round(Tr[0], 3).tolist())
print("same as Chapter 6's loops:", bool(np.allclose(Tr, agreement_table(Xr))))

tok = load_bpe(TINY_BPE)
B = Embedding(TINY_VOCAB_SIZE, seed=1865)(tok.encode(SENTENCE))   # Tiny's block, 7 x 128
TB = B @ B.T
print(f"\nTiny: block {B.shape} @ block.T {B.T.shape} -> table {TB.shape}")
print("first row:", "[" + ", ".join(f"{v:.4f}" for v in TB[0]) + "]")
print(f"table sum: {TB.sum():.4f}; same as the loops: {bool(np.allclose(TB, agreement_table(B)))}")
