"""Chapter 9, section 1: one blend is not enough.  Mat needs two things from its sentence: the verb
(sat) and the word just before it (on).  A single head must put both into one row of shares, and the
blend of sat's value and on's value is (1, 0), which is exactly the value of 'the'.  Two heads, each
with its own small tables, fetch one each and keep them apart.  The tables are found by a small,
exhaustive search over whole numbers from -2 to 2, so that every number can be checked by hand.
Run from 04-code/ch09:  python3 one_vs_two.py"""
from _common import *
from itertools import product
from tiny.attention import attention

V_id = np.eye(2)                                  # values = the token rows themselves, to keep the arithmetic visible
grid = [np.array(t, dtype=float).reshape(2, 2) for t in product(range(-2, 3), repeat=4)]

def mat_row(Wq, Wk):
    _, A = attention(X, Wq, Wk, V_id)
    return A[5]

def best(score):
    top, arg = -1, None
    for Wq in grid:
        for Wk in grid:
            s = score(mat_row(Wq, Wk))
            if s > top + 1e-12:
                top, arg = s, (Wq, Wk)
    return arg

print("values are the toy rows: sat", X[2].tolist(), " on", X[3].tolist(), " the", X[0].tolist())
print("a 50/50 blend of sat and on:", (0.5 * X[2] + 0.5 * X[3]).tolist(), "<- the same row as 'the'")
qa, ka = best(lambda r: r[2])                      # head A: mat -> sat
qb, kb = best(lambda r: r[3])                      # head B: mat -> on
qs, ks = best(lambda r: min(r[2], r[3]))           # one head asked to fetch both
for name, (Wq, Wk) in (("head A (wants sat)", (qa, ka)), ("head B (wants on)", (qb, kb)), ("one head for both", (qs, ks))):
    out, A = attention(X, Wq, Wk, V_id)
    print(f"\n{name}: query table {Wq.astype(int).tolist()} key table {Wk.astype(int).tolist()}")
    print("  mat's shares:", [round(float(v), 4) for v in A[5]], "  mat's output:", [round(float(v), 4) for v in out[5]])
outA, _ = attention(X, qa, ka, V_id); outB, _ = attention(X, qb, kb, V_id); outS, _ = attention(X, qs, ks, V_id)
print("\ntwo heads side by side, mat:", r3(np.concatenate([outA[5], outB[5]])))
d = lambda a, b: float(np.linalg.norm(a - b))
print(f"distance of the single head's output from sat {d(outS[5], X[2]):.3f}, from on {d(outS[5], X[3]):.3f}, from the {d(outS[5], X[0]):.3f}")
print(f"distance of head A's output from sat {d(outA[5], X[2]):.3f}; of head B's from on {d(outB[5], X[3]):.3f}")
