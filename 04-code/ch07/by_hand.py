"""Section 7.1: the 6 x 6 agreement table of the plain word rows, cell by cell, with Chapter 6's loops.
Run from 04-code/ch07:  python3 by_hand.py"""
import _common
import numpy as np
from tiny.tokenizer import TOY_VOCAB
from tiny.embed import TOY_EMBED, dot, agreement_table

words = "the cat sat on the mat".split()
X = TOY_EMBED[[TOY_VOCAB.index(w) for w in words]]          # six plain word rows, two numbers each

# One row of the table by hand: sat asking every word
sat = X[2]
p = lambda v: f"({v:g})" if v < 0 else f"{v:g}"              # show negatives in brackets
for w, row in zip(words, X):
    a, b = sat[0] * row[0], sat[1] * row[1]
    print(f"sat . {w:<3} = {p(sat[0])} x {p(row[0])} + {p(sat[1])} x {p(row[1])} = {p(a)} + {p(b)} = {a + b:g}")

T = agreement_table(X)                                       # Chapter 6: two loops, 36 calls of dot
print("\n       " + "".join(f"{w:>5}" for w in words))
for w, r in zip(words, T):
    print(f"{w:>5}  " + "".join(f"{v:5g}" for v in r))

n, d = X.shape
print(f"\ndot products: {n} x {n} = {n * n}; multiplications {n * n * d}, additions {n * n * (d - 1)}")
print("mirror image across the diagonal (row i, col j = row j, col i):", bool((T == T.T).all()))
print("diagonal = each row's squared length:", T.diagonal().tolist())
print("different numbers you actually need to compute:", n * (n + 1) // 2)
