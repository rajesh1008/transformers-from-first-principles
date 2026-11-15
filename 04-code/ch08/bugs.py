"""Chapter 8: six bugs that run.  Each is one changed line in the twelve-line head; each runs
without an error message.  For each: sat's row, the row sums, and whether the leak test passes.
Run from 04-code/ch08:  python3 bugs.py"""
from _common import *

def head(X_, bug=None, d_scale=None):
    Q, K, V = X_ @ W_q, X_ @ W_k, X_ @ W_v
    S = (K @ Q.T if bug == "swapped" else Q @ K.T) / np.sqrt(d_scale or K.shape[1])
    n = len(X_)
    mask = np.triu(np.ones((n, n), dtype=bool)) if bug == "triu" else np.tril(np.ones((n, n), dtype=bool))
    if bug != "after":
        S = np.where(mask, S, -np.inf)
    axis = 0 if bug == "axis0" else 1
    e = np.exp(S - S.max(axis=axis, keepdims=True))
    A = e / (e.sum(axis=1) if bug == "nokeep" else e.sum(axis=axis, keepdims=True))
    if bug == "after":
        A = np.where(mask, A, 0.0)
    return A @ V, A

def leaks(bug):
    X2 = X.copy(); X2[5] = [-2, 1]
    a, _ = head(X, bug); b, _ = head(X2, bug)
    return int(sum(not np.allclose(a[i], b[i]) for i in range(5)))

cases = [(None, "correct"),
         ("axis0", "softmax down the columns (axis=0) instead of along the rows"),
         ("nokeep", "row totals without keepdims=True: e / e.sum(axis=1)"),
         ("after", "mask applied after softmax instead of before"),
         ("triu", "mask built with np.triu (upper triangle) instead of np.tril"),
         ("swapped", "K @ Q.T instead of Q @ K.T")]
for bug, name in cases:
    out, A = head(X, bug)
    print(f"{name}")
    print(f"   sat's row {np.round(A[2], 3).tolist()}  output {tuple(np.round(out[2], 3).tolist())}")
    print(f"   row sums {np.round(A.sum(axis=1), 3).tolist()}  earlier rows that change when mat->hat: {leaks(bug)} of 5")

print("\nscaled by the wrong width: a head of width 32 inside a model of width 128 (seed 1865, 2,000 rows of 6)")
rng = np.random.default_rng(1865)
q, k = rng.normal(size=(2000, 32)), rng.normal(size=(2000, 6, 32))
raw = np.einsum("nd,nkd->nk", q, k)
from tiny.attention import softmax, entropy
for name, div in (("sqrt(32), the head's width", np.sqrt(32)), ("sqrt(128), the model's width", np.sqrt(128))):
    sh = softmax(raw / div)
    print(f"   divide by {name:<28}: top share {sh.max(1).mean():.3f}  entropy {entropy(sh).mean():.3f}")
