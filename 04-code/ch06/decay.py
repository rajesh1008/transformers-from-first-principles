"""Does rotary agreement fade with distance?  For a row with every hand at 12 o'clock, exactly Chapter 5's curve.
For random rows, the gap changes the score, but there is no fixed fade.  Run: python3 decay.py"""
import _common
import numpy as np
from tiny.embed import apply_rotary, sinusoidal, dot, clock_speeds

d = 128
up = np.array([[0.0, 1.0] * (d // 2)])           # every pair = a hand pointing at 12 o'clock
S = sinusoidal(5, d)
print("rotary applied to (0, 1, 0, 1, ...) at position 4 == sinusoidal row 4:",
      bool(np.allclose(apply_rotary(up, [4])[0], S[4])))
speeds = clock_speeds(d)
gaps = [0, 1, 10, 100, 1000]
for g in gaps:
    a, b = apply_rotary(up, [0])[0], apply_rotary(up, [g])[0]
    print(f"  all-up row, gap {g:>4}: agreement {dot(a, b):7.2f}")

rng = np.random.default_rng(1865)
Q = rng.normal(size=(400, d)); K = rng.normal(size=(400, d))
print("400 random query/key pairs (each number spread 1): average agreement and average size by gap")
for g in gaps:
    s = [dot(apply_rotary([q], [0])[0], apply_rotary([k], [g])[0]) for q, k in zip(Q, K)]
    same = [dot(apply_rotary([q], [0])[0], apply_rotary([q], [g])[0]) for q in Q]
    print(f"  gap {g:>4}: random pair average {np.mean(s):6.2f}, average size {np.mean(np.abs(s)):5.2f};"
          f"  a row with itself {np.mean(same):6.2f}")
