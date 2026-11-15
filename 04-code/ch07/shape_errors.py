"""Section 7.6: reading a shape error.  Three loud mistakes and two silent ones, with NumPy's real words.
Run from 04-code/ch07:  python3 shape_errors.py"""
import _common
import numpy as np
from tiny.tokenizer import TOY_VOCAB
from tiny.embed import TOY_EMBED, TOY_POS

X = TOY_EMBED[[TOY_VOCAB.index(w) for w in "the cat sat on the mat".split()]]     # (6, 2)

def attempt(label, f):
    try:
        out = f()
        print(f"{label}\n   no error; answer has shape {np.shape(out)}")
        return out
    except Exception as e:
        print(f"{label}\n   {type(e).__name__}: {e}")

print("LOUD")
attempt("1. X @ X            (forgot to turn the second one on its side)", lambda: X @ X)
attempt("2. X + TOY_POS[:3]  (three position rows for six tokens)", lambda: X + TOY_POS[:3])
attempt("3. X @ np.ones(3)   (a list of the wrong length)", lambda: X @ np.ones(3))

print("\nSILENT")
G = attempt("4. X.T @ X          (turned the wrong one)", lambda: X.T @ X)
print("   it is a real table, columns against columns:", G.tolist())
lengths = (X * X).sum(axis=1)                   # shape (6,): one squared length per token
column = lengths.reshape(6, 1)                  # the same six numbers, stood upright: (6, 1)
D = attempt("5. lengths - column (one list flat, one upright)", lambda: lengths - column)
print("   wanted six zeros; got", D.shape[0] * D.shape[1], "numbers, of which", int((D != 0).sum()), "are not zero")
