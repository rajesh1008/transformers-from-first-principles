"""Appendix C, sections C.8 and C.9: the running sentence's 37,001,216 FLOPs through Tiny (Chapter 16's count)
split by kind of product, and the same products' backward-pass count by Chapter 17 and 18's rule (each product
Y = X W costs 2 forward and 2 + 2 backward per multiply-add; attention's two n x n x d products become four).
Arithmetic only; Tiny's shape: 7 tokens, width 128, 12 layers, feed-forward 512, vocabulary 2,048."""
import _common
n, d, L, f, V = 7, 128, 12, 512, 2048
parts = {"query, key, value and output tables": L * 4 * n * d * d,
         "attention scores and blend (two n x n x d products)": L * n * n * d * 2,
         "feed-forward network": L * 2 * n * d * f,
         "output scores (tied token table)": n * d * V}
tot = 0
print("forward FLOPs (2 per multiply-add) for 'the cat sat on the mat', 7 tokens through Tiny:")
for k, v in parts.items():
    print(f"  {k:<52} {2 * v:>12,}"); tot += 2 * v
print(f"  {'total':<52} {tot:>12,}  ({tot // n:,} per token)")
assert tot == 37001216
back = 2 * tot
print(f"backward pass, twice the forward products (dX = dY W^T and dW = X^T dY; attention 4 against 2): {back:,}")
print(f"one training step on the sentence, forward plus backward: {tot + back:,} FLOPs, 3 x the forward")
