"""Math Interlude: a matrix as a stack of lists; rows, columns, shape, turning on its side;
one product computed entry by entry.   Run from 04-code/ch07:  python3 stack.py"""
import _common
import numpy as np
from tiny.tokenizer import TOY_VOCAB
from tiny.embed import TOY_EMBED

words = "the cat sat on the mat".split()
X = TOY_EMBED[[TOY_VOCAB.index(w) for w in words]]     # the stack: one row per token
print("X has shape", X.shape, "(rows, columns)")
print("row 2 (sat):", X[2].tolist(), "  column 0 (every first number):", X[:, 0].tolist())
print("entry in row 5, column 0 (mat's first number):", X[5, 0])
print("X turned on its side, X.T, has shape", X.T.shape, "; its row 0 is X's column 0:", X.T[0].tolist())

A = np.array([[1, 1], [-1, 1]])        # two asking rows: sat, mat
B = np.array([[0, 1], [1, -1]])        # two answering COLUMNS: cat (0, 1) and on (1, -1), stood upright
print("\nA (rows sat, mat) =", A.tolist(), "  B (columns cat, on) =", B.tolist())
for i, who in enumerate(["sat", "mat"]):
    for j, whom in enumerate(["cat", "on"]):
        terms = " + ".join(f"{A[i, k]} x {B[k, j]}" for k in range(2))
        print(f"  entry ({i}, {j}) = {who} . {whom} = {terms} = {A[i] @ B[:, j]}")
print("A @ B =", (A @ B).tolist())
print("B @ A =", (B @ A).tolist(), "  (order matters)")

v = X[2:3]                              # sat as a stack of one row: shape (1, 2)
w = X[1:2].T                            # cat stood upright: shape (2, 1)
print("\none dot product as a product of shapes", v.shape, "and", w.shape, "->", (v @ w).shape, "=", (v @ w).tolist())
G = X.T @ X                             # the other way round: (2, 6) @ (6, 2)
print("X.T @ X has shape", G.shape, "=", G.tolist(), " (columns agreeing with columns)")
