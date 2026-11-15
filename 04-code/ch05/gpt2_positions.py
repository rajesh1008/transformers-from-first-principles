"""Section 5.3: what GPT-2 learned for its 1,024 positions.  Run: python3 gpt2_positions.py"""
import _common
import numpy as np
import _gpt2
from tiny.embed import cosine, length

P = _gpt2.tensor("wpe.weight")
E = _gpt2.tensor("wte.weight")
print("GPT-2 position table:", P.shape, f"= {P.size:,} learned numbers")
L = [length(r) for r in P]
print("row lengths, positions 0-5:", [round(float(x), 2) for x in L[:6]])
print(f"median row length: {np.median(L):.2f}; median token-row length: {np.median([length(r) for r in E]):.2f}")
print("longest row: position", int(np.argmax(L)), "; shortest row: position", int(np.argmin(L)), f"(length {min(L):.3f})")
U = P / np.array(L)[:, None]
C = U @ U.T          # every cosine at once; a measuring tool here, taught in Chapter 7
for k in (1, 10, 50, 100, 200, 500, 1000):
    print(f"average cosine of positions {k:>4} apart: {np.mean([C[i, i + k] for i in range(1024 - k)]):6.3f}")
first = next(k for k in range(1, 1024) if np.mean([C[i, i + k] for i in range(1024 - k)]) < 0.5)
print("first gap at which the average cosine falls below 0.5:", first)
print("cosine(position 0, 1):", round(float(C[0, 1]), 3), "; cosine(position 1, 2):", round(float(C[1, 2]), 3))
the = E[262]          # ' the'
print("' the' at position 1 vs 5: cosine", round(cosine(the + P[1], the + P[5]), 3),
      "; at 1 vs 900:", round(cosine(the + P[1], the + P[900]), 3))
# wave-like columns: count how often a column changes sign as the position runs 1..1022
cols = np.argsort(-P[1:1023].std(axis=0))[:4]
for c in cols:
    col = P[1:1023, c]
    print(f"column {c}: spread {col.std():.2f}, sign changes along the positions {int((np.sign(col[1:]) != np.sign(col[:-1])).sum())}")
