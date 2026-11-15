"""Section 7.3: NumPy as a calculator.  Star versus @; adding along a direction; one row added to
every row (broadcasting); Chapter 6's two turning formulas as one small matrix.
Run from 04-code/ch07:  python3 numpy_calc.py"""
import _common
import math
import numpy as np
from tiny.tokenizer import TOY_VOCAB
from tiny.embed import TOY_EMBED, TOY_POS, TOY_TURN, add_positions, apply_rotary

X = TOY_EMBED[[TOY_VOCAB.index(w) for w in "the cat sat on the mat".split()]]
sat, cat = X[2], X[1]
print("sat * cat  (entry by entry):", (sat * cat).tolist(), "  sum:", (sat * cat).sum())
print("sat @ cat  (the dot product):", sat @ cat)
print("X * X has shape", (X * X).shape, "; X @ X.T has shape", (X @ X.T).shape)
print("sum of each row of X * X  (axis=1):", (X * X).sum(axis=1).tolist(), "= the table's diagonal")
print("sum down each column of X (axis=0):", X.sum(axis=0).tolist())

S = X + TOY_POS                                   # Chapter 5's whole add_positions, one symbol
print("\nX + TOY_POS, row sat:", S[2].tolist(), "  same as Chapter 5's loops:",
      bool(np.array_equal(S, add_positions(X, TOY_POS))))
shift = np.array([0.5, 0.5])
print("X + one row (0.5, 0.5): shapes", X.shape, "+", shift.shape, "->", (X + shift).shape,
      "; row mat:", (X + shift)[5].tolist())

def turn_matrix(angle):
    """Chapter 6's two formulas, written as a stack of two lists."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, s],      # new across = across*cos + up*sin
                     [-s, c]])    # new up     = up*cos - across*sin
R = turn_matrix(2 * TOY_TURN[0])                  # seat 2: 60 degrees
print("\nturn by 60 degrees as a matrix:", np.round(R, 3).tolist())
print("sat turned, R @ sat:", np.round(R @ sat, 3).tolist(), "  Chapter 6:",
      np.round(apply_rotary(X[2:3], [2], speeds=TOY_TURN)[0], 3).tolist())
