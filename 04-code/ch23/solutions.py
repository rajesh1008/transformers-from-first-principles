"""Appendix A, Chapter 23: the Check Yourself box and by-hand Exercises 1 to 7, computed."""
import _common
import math
import numpy as np
from tiny import scaling as S
print("Check yourself (Interlude 23.3)")
A, a = 12.0, 0.25
for n in (1e4, 2e4, 1.6e5):
    print(f"  L = 12 N^-0.25 at N = {n:,.0f}: {A * n ** -a:.4f}")
print(f"  each doubling multiplies by 2^-0.25 = {2 ** -0.25:.4f}; log-log slope -0.25; "
      f"16x more N divides by {16 ** 0.25:.1f}")
print(f"  with a floor 4: L = 4 + 12 N^-0.25 at 1e4 {4 + 12 * 1e4 ** -0.25:.4f}, at 1.6e5 {4 + 12 * 1.6e5 ** -0.25:.4f}; the straight-line slope between them "
      f"{-(math.log(4 + 12 * 1.6e5 ** -0.25) - math.log(4 + 12 * 1e4 ** -0.25)) / math.log(16):.4f}")
print("\nExercise 1")
print(f"  2^-0.3 = {2 ** -0.3:.4f}; doublings to halve the reducible loss 1/0.3 = {1 / 0.3:.2f}; size factor {2 ** (1 / 0.3):.1f}")
print("Exercise 2")
al = math.log(3.0 / 2.0) / math.log(8.0)
print(f"  alpha = ln(3/2)/ln 8 = {al:.4f}; A = 3.0 x (1e5)^alpha = {3.0 * 1e5 ** al:.3f}")
print("Exercise 3")
C = 6 * 791_296 * 393_216
print(f"  6ND = {C:.4e}; at 38 G per second {C / 38e9:.1f} s")
print("Exercise 4")
Cb = 1e21
Nopt = math.sqrt(Cb / (6 * 20)); print(f"  N = sqrt(C / 120) = {Nopt:.3e}, D = 20 N = {20 * Nopt:.3e}")
print("Exercise 5")
a_, b_, c_ = 5.6, 5.3, 5.1
r = (b_ - c_) / (a_ - b_); alpha = -math.log2(r); t = (a_ - b_) / (1 - r)
print(f"  ratio of drops r = {r:.4f}; alpha = -log2 r = {alpha:.4f}; A N^-alpha at N = {t:.4f}; E = {a_ - t:.4f}")
x = np.array([1e5, 2e5, 4e5]); f = S.fit_floor(x, [a_, b_, c_])
print(f"  check with tiny/scaling.fit_floor: E {f['E']:.4f} alpha {f['alpha']:.4f}, misses {f['sse']:.1e}")
print("Exercise 6")
s, d = 0.15, 0.10
print(f"  k >= (2 s / d)^2 = {(2 * s / d) ** 2:.1f}, so 9 orders")
print("Exercise 7")
dmin = 9 * 2176 / 48
print(f"  share = 2176 d / (2176 d + 48 d^2) < 0.1 when d > {dmin:.0f}; at d = 128: {2176 / (2176 + 48 * 128):.3f} (biases and norms ignored)")
