"""Section 5.4: Tiny's sinusoidal rows, and the three facts that make them useful.  Run: python3 sinusoidal.py"""
import _common
import time
import numpy as np
from tiny.embed import sinusoidal, dot, length, cosine, TINY_CONTEXT, TINY_D

t0 = time.perf_counter(); S = sinusoidal(TINY_CONTEXT, TINY_D); t = time.perf_counter() - t0
print("sinusoidal rows:", S.shape, f"computed with plain loops in {t * 1000:.0f} ms; learned numbers: 0")
print("position 4 starts", np.round(S[4][:6], 4).tolist())
print("fact 1, every row has the same length:", sorted({float(round(length(r), 9)) for r in S}), "= square root of", TINY_D // 2)
print("fact 2, agreement depends only on the gap:")
for a in (0, 100, 1000):
    print(f"  dot(position {a:>4}, position {a + 4:>4}) = {dot(S[a], S[a + 4]):.4f}")
print("  gap 1:", round(dot(S[0], S[1]), 4), " gap 10:", round(dot(S[0], S[10]), 4), " gap 100:", round(dot(S[0], S[100]), 4),
      " gap 1000:", round(dot(S[0], S[1000]), 4))
U = S / 8.0
C = U @ U.T
np.fill_diagonal(C, -2)
print("fact 3, no two of the 1,024 rows are the same: closest pair has cosine", round(float(C.max()), 4),
      "(positions", tuple(int(i) for i in np.unravel_index(C.argmax(), C.shape)), ")")
for k in (1, 10, 50, 100, 200, 500, 1000):
    print(f"average cosine of positions {k:>4} apart: {np.mean([C[i, i + k] for i in range(TINY_CONTEXT - k)]):6.3f}")
