"""Chapter 35, 35.3: SmolLM2-135M's rotary pairs by hand.  Head width 64 (576 / 9), so 32 pairs; pair i turns by
base^(-2i/64) radians per token.  For each base, the period of the fastest and slowest pair, and how many pairs never complete
one turn inside 2,048, 8,192 and 16,384 tokens; and the largest angle the slowest pair reaches at each length."""
import math, json, os
hd = 64
for base in (100000, 400000, 1600000):
    inv = [base ** (-2 * i / hd) for i in range(hd // 2)]
    per = [2 * math.pi / w for w in inv]
    print(f"base {base:,}: fastest pair period {per[0]:.2f} tokens, slowest {per[-1]:,.0f} tokens")
    for L in (2048, 8192, 16384):
        n = sum(p > L for p in per)
        print(f"   inside {L:6,} tokens: {n:2d} of 32 pairs never complete a turn; slowest pair reaches {L * inv[-1]:.3f} radians")
