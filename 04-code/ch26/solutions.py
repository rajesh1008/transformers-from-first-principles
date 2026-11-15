"""Appendix A, Chapter 26: Check Yourself and Exercises 1 to 5 computed (6 and 7 are answered in words in Appendix A)."""
import math
import numpy as np
print("Check yourself")
print(f"  1  sqrt(0.08^2 + 0.06^2) = {math.sqrt(0.08 ** 2 + 0.06 ** 2):.4f}")
print(f"  2  0.1 / sqrt(k) = 0.02  ->  k = {(0.1 / 0.02) ** 2:.0f}")
print(f"  3  0.1 x sqrt(2) = {0.1 * math.sqrt(2):.4f}")
print("Exercises")
f = np.float32
print(f"  1  (1e8 + 3) - 1e8 = {float((f(1e8) + f(3)) - f(1e8))}; (1e8 + 5) - 1e8 = {float((f(1e8) + f(5)) - f(1e8))}; gap at 1e8 = {float(np.spacing(f(1e8)))}")
g = float(np.spacing(f(320)))
print(f"  2  gap between 32-bit numbers near 320 = {g:.3g}; the left-to-right error 0.010360 is {0.010360 / g:.0f} gaps")
print(f"  3  sqrt(0.12^2 + 0.05^2) = {math.sqrt(0.12 ** 2 + 0.05 ** 2):.4f}")
print(f"  4  spread 0.04: three seeds {0.04 / math.sqrt(3):.4f}, five seeds {0.04 / math.sqrt(5):.4f}")
sd = 0.03 * math.sqrt(2); z = 0.02 / sd
p = 0.5 * math.erfc(z / math.sqrt(2))
print(f"  5  spread of a difference of single runs {sd:.4f}; true difference 0.02 is {z:.2f} spreads; chance of the wrong sign {p:.2f}")
k = round((0.03 * math.sqrt(2) / 0.01) ** 2)
print(f"     seeds per recipe for the difference's spread to be 0.01 (half the effect): {k}")
