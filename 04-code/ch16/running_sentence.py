"""The running sentence's cost through Tiny, by hand: seven tokens under Tiny's vocabulary, the same
multiply-add count as cost.py (Tiny's code computes the full score table and masks it)."""
from _common import c
n, d, L, f, V = 7, 128, 12, 512, 2048
madds = L * 4 * n * d * d + L * n * n * d * 2 + L * 2 * n * d * f + n * d * V
print(f"the cat sat on the mat, 7 tokens through Tiny: {c(2 * madds)} FLOPs, {c(2 * madds // n)} per token")
