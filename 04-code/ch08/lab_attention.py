"""Lab 8: one full head in NumPy, then break it twice.
Rows: Chapter 4's toy rows plus Chapter 5's toy positions.  Tables: 2 x 2, drawn with seed 1865.
Run from 04-code/ch08:  python3 lab_attention.py"""
from _common import *
from tiny.attention import attention, softmax, causal_mask, entropy
from tiny.embed import TOY_POS

SEED = 1865
rng = np.random.default_rng(SEED)
Wq, Wk, Wv = (np.round(rng.normal(size=(2, 2)), 2) for _ in range(3))   # two decimals, pencil-friendly
R = X + TOY_POS                                                          # (6, 2)
print("seed", SEED)
print("query table", Wq.tolist(), " key table", Wk.tolist(), " value table", Wv.tolist())
print("rows with positions", R.tolist())

out, A = attention(R, Wq, Wk, Wv)
print("\nSTEP 1  the Attention Table (masked, scaled shares; rows ask, columns answer)")
print(table(A))
print("  each row sums to 1:", bool(np.allclose(A.sum(axis=1), 1)), "| entropy per row:", np.round(entropy(A), 3).tolist())
print("STEP 2  outputs")
for w, o in zip(WORDS, out):
    print(f"  {w:>4} {tuple(round(float(v), 3) for v in o)}")

print("\nBREAKAGE 1: remove the scaling (Tiny's width, 128; rows with spread 1, seed 1865)")
r2 = np.random.default_rng(SEED)
Xb = r2.normal(size=(6, 128)); Wq_b, Wk_b, Wv_b = (r2.normal(size=(128, 128)) / np.sqrt(128) for _ in range(3))
for scale in (True, False):
    _, Ab = attention(Xb, Wq_b, Wk_b, Wv_b, scale=scale)
    print(f"  scale {'on ' if scale else 'off'}: sat's row {np.round(Ab[2, :3], 3).tolist()}  mat's row {np.round(Ab[5], 3).tolist()}  rows with a share above 0.99: {int((Ab.max(axis=1) > 0.99).sum())} of 6")

print("\nBREAKAGE 2: remove the mask")
_, A_open = attention(R, Wq, Wk, Wv, causal=False)
print("  share the first 'the' gives to 'mat' (five places ahead):", round(float(A_open[0, 5]), 3), "| with the mask:", round(float(A[0, 5]), 3))
print("  shares above the diagonal, total:", round(float(np.triu(A_open, 1).sum()), 3), "| with the mask:", round(float(np.triu(A, 1).sum()), 3))

print("\nCHECKPOINT")
print("  sat's row:", [round(float(v), 3) for v in A[2]])
print("  mat's row:", [round(float(v), 3) for v in A[5]])
print("  table sum 6.000 and trace", round(float(np.trace(A)), 3))
