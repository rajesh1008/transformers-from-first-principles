"""Lab 9: four heads on the running sentence, then break them twice.
Rows: Chapter 4's toy rows plus Chapter 5's toy positions (6 x 2).  Four heads, each with its own
2 x 2 query, key and value tables, drawn with seed 1865 in the order head 0 (query, key, value),
head 1, head 2, head 3, rounded to two decimals; head 0 is therefore Lab 8's head.  The four
outputs (6 x 2 each) are put side by side (6 x 8) and multiplied by an 8 x 2 output table drawn
next from the same stream.  Run from 04-code/ch09:  python3 lab_multihead.py"""
from _common import *
from tiny.attention import attention, entropy, MultiHead

rng = np.random.default_rng(SEED)
heads = [tuple(np.round(rng.normal(size=(2, 2)), 2) for _ in range(3)) for _ in range(4)]
W_o = np.round(rng.normal(size=(8, 2)), 2)
R = XP
print("seed", SEED, "| rows with positions", R.tolist())
outs, tabs = [], []
for h, (q, k, v) in enumerate(heads):
    out, A = attention(R, q, k, v)
    outs.append(out); tabs.append(A)
    print(f"\nhead {h}: query {q.tolist()}  key {k.tolist()}  value {v.tolist()}")
    print(table(A))
    print(f"  trace {np.trace(A):.3f} | entropy of mat's row {entropy(A)[5]:.3f} | sat's favourite: {WORDS[int(np.argmax(A[2]))]}")
side = np.concatenate(outs, axis=1)
print("\noutput table W_o (8 x 2):", W_o.tolist())
final = side @ W_o
print("side by side (6 x 8), sat's row:", r3(side[2]))
print("after the output table (6 x 2):")
for w, o in zip(WORDS, final):
    print(f"  {w:>4} {r3(o)}")
print("the same as adding each head's output times its own two rows of W_o:",
      bool(np.allclose(final, sum(o @ W_o[2 * h:2 * h + 2] for h, o in enumerate(outs)))))
print("favourite seat for each row, per head:")
for h, A in enumerate(tabs):
    print(f"  head {h}: " + " ".join(WORDS[int(j)] for j in A.argmax(1)))

print("\nBREAKAGE 1: give every head the same tables (head 0's)")
same = [attention(R, *heads[0]) for _ in range(4)]
tabs_s = [A for _, A in same]
print("  distinct Attention Tables among the four:", len({np.round(A, 9).tobytes() for A in tabs_s}))
side_s = np.concatenate([o for o, _ in same], axis=1)
merged = W_o[0:2] + W_o[2:4] + W_o[4:6] + W_o[6:8]
print("  four identical heads with W_o equal one head with the four blocks of W_o added:",
      bool(np.allclose(side_s @ W_o, same[0][0] @ merged)), " merged output table", np.round(merged, 2).tolist())
print("  learned numbers in the four heads: 4 x 12 = 48; numbers that make a difference: 12, plus 4 of output table")
print(f"  width of the side-by-side rows 8, but they span only {np.linalg.matrix_rank(side_s)} directions"
      f" (with four different heads: {np.linalg.matrix_rank(side)})")

print("\nBREAKAGE 2: remove a head in the wrong place (Tiny's width, four heads of 32, rows with spread 1, seed 1865)")
m = MultiHead(n_heads=4, spread=1.0 / np.sqrt(128))
Xb = np.random.default_rng(SEED).normal(size=(6, 128))
full = m(Xb)
right = m(Xb, ablate=[0])                                   # head 0's output set to zero before the output table
naive = full.copy(); naive[:, :32] = 0.0                    # 32 columns zeroed after the output table
o, _ = m.head_outputs(Xb)
contrib0 = o[0] @ m.W_o[:32]                                 # what head 0 really added
print(f"  head 0's contribution, size {np.linalg.norm(contrib0):.3f}; the correct removal takes away exactly that:",
      bool(np.allclose(full - right, contrib0)))
removed = full - naive
print(f"  the naive removal takes away {np.linalg.norm(removed):.3f} of output, of which head 0's part is"
      f" {np.linalg.norm(contrib0[:, :32]):.3f}; it leaves {100 * np.linalg.norm(contrib0[:, 32:]) ** 2 / np.linalg.norm(contrib0) ** 2:.0f}"
      f" per cent of head 0's contribution in place")
c = np.corrcoef(removed[:, :32].ravel(), contrib0[:, :32].ravel())[0, 1]
print(f"  correlation between what was removed and head 0's own part of those columns: {c:.2f}")

print("\nCHECKPOINT")
for h, A in enumerate(tabs):
    print(f"  head {h}: sat's row {r3(A[2, :3])}  trace {np.trace(A):.3f}")
print("  sat's final output", r3(final[2]))
