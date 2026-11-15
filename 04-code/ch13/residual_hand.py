"""Chapter 13, section 1: the residual connection as 'keep what you had', on the toy rows.
Chapter 8's hand head and Chapter 12's hand block, stacked six times with the same tables, with and
without the additions, and with RMSNorm in front of each (pre-norm; layer norm cannot be used on two
numbers, as hand_norm.py shows).  Then the residual stream as bookkeeping: the top row is the bottom row
plus every block's output, exactly.  Run from 04-code/ch13:  python3 residual_hand.py"""
from _common import *
from tiny.norm import rms_norm, token_similarity

mat = 5
out = hand_head(X)
print("one head on the toy rows")
print(f"  mat's row {r3(X[mat])}; the head brings {r3(out[mat])}")
print(f"  with the addition (x + head(x)): {r3(X[mat] + out[mat])} | without (head(x) alone): {r3(out[mat])}")
print(f"  cosine of mat's new row with its old row: with {np.dot(X[mat], X[mat] + out[mat]) / np.linalg.norm(X[mat]) / np.linalg.norm(X[mat] + out[mat]):.3f}, "
      f"without {np.dot(X[mat], out[mat]) / np.linalg.norm(X[mat]) / np.linalg.norm(out[mat]):.3f}")

def run(wiring, n=6):
    x = X.copy(); rows = []
    for _ in range(n):
        for f in (hand_head, hand_block):
            if wiring == "none":
                x = f(x)
            elif wiring == "residual":
                x = x + f(x)
            elif wiring == "pre":
                x = x + f(rms_norm(x))
        rows.append(x.copy())
    return rows

print("\nthe hand layer (head, then block) six times, the same tables each time:")
for wiring in ("none", "residual", "pre"):
    rows = run(wiring)
    print(f"  {wiring}:")
    for i, x in enumerate(rows, 1):
        if i in (1, 2, 3, 6):
            sim = token_similarity(x) if np.abs(x).sum() > 0 else float("nan")
            print(f"    after {i}: mat {r3(x[mat])} | sat {r3(x[2])} | mean length {np.linalg.norm(x, axis=1).mean():.3f} | similarity {sim:.3f}")
    xs = rows[-1]
    print(f"    after 6, words with the same row as 'the' (to 3 places): {[WORDS[j] for j in range(6) if np.allclose(np.round(xs[j], 3), np.round(xs[0], 3))]}")

# the stream as bookkeeping
x = X.copy(); writes = []
for _ in range(3):
    for f in (hand_head, hand_block):
        w = f(rms_norm(x)); writes.append(w); x = x + w
print("\nthe residual stream after three pre-norm hand layers, for mat:")
print(f"  top row {r3(x[mat])} = bottom row {r3(X[mat])} + six writes {[r3(w[mat]) for w in writes]}")
print(f"  sum check: {r3(X[mat] + sum(w[mat] for w in writes))}; exact: {bool(np.allclose(x, X + sum(writes)))}")
