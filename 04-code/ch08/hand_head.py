"""Chapter 8: one head of attention on the running sentence, every step printed.
Toy rows from Chapter 4; query and key tables from the Voice Reference; a value table added here.
Run from 04-code/ch08:  python3 hand_head.py"""
from _common import *
from tiny.attention import softmax, causal_mask

Q, K, V = X @ W_q, X @ W_k, X @ W_v
print("STEP 1  three rows per token")
for w, q, k, v in zip(WORDS, Q, K, V):
    print(f"  {w:>4}  query {tuple(int(a) for a in q)!s:>8}  key {tuple(int(a) for a in k)!s:>8}  value {tuple(int(a) for a in v)!s:>8}")

S = Q @ K.T
print("\nSTEP 2  scores, Q @ K.T (rows ask, columns answer)")
print(table(S, fmt="{:>7.0f}"))
print("  mirror image?", bool(np.allclose(S, S.T)), "| sat->cat", S[2, 1], " cat->sat", S[1, 2])

Ss = S / np.sqrt(2)
print("\nSTEP 3  scaled scores, divided by sqrt(2) = 1.414")
print(table(Ss))

print("\nSTEP 4  softmax on sat's row, by hand")
row = Ss[2]
e = np.exp(row)
for w, s, x in zip(WORDS, row, e):
    print(f"  {w:>4}  score {s:6.3f}  exp {x:7.3f}  share {x / e.sum():.3f}")
print(f"  total of the exponentials {e.sum():.3f}; shares sum to {(e / e.sum()).sum():.3f}")

A_open = softmax(Ss)
print("\nSTEP 5  shares with no mask (every row sums to one)")
print(table(A_open))

M = causal_mask(6)
A = softmax(np.where(M, Ss, -np.inf))
print("\nSTEP 6  the causal mask: shares when each token sees only itself and earlier tokens")
print(table(A))
print("  row sums:", [round(float(s), 3) for s in A.sum(axis=1)])

out = A @ V
print("\nSTEP 7  weighted sums of the values (masked shares @ V)")
for w, o, x in zip(WORDS, out, X):
    print(f"  {w:>4}  was {tuple(float(a) for a in x)!s:>12}  now holds {tuple(round(float(a), 3) for a in o)}")

print("\n  sat's output, piece by piece:")
for j in range(3):
    print(f"    {A[2, j]:.3f} x value of {WORDS[j]:>3} {tuple(int(a) for a in V[j])} = {tuple(round(float(a), 3) for a in A[2, j] * V[j])}")
print(f"    sum = {tuple(round(float(a), 3) for a in out[2])}")
print("  share of cat in sat's row:", round(float(A[2, 1]), 3), "(no mask:", round(float(A_open[2, 1]), 3), ")")
