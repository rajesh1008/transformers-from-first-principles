"""Chapter 22: numerical answers to the by-hand exercises 1-7 (seed 1865).  Exercises 8-12 are code exercises
whose starting points are train_data.py, tiny/data.py and stream.py; they were not run for this edition."""
import math
import numpy as np
from _common import SEED

rng = np.random.default_rng(SEED)
# 1. padding vs packing, documents of 300 tokens, windows of 4,097 tokens (4,096 predictions)
L, W = 300, 4096
pad_waste = 1 - (L - 1) / W
starts = rng.integers(0, L, 20000)                     # a window starts at a uniform place inside a document
first = np.minimum(L - starts, W)                      # predictions that see only the first (cut) document
print(f"1. padding wastes {pad_waste:.1%} of seats; packed: {1 - first.mean() / W:.1%} of predictions see another document"
      f" (exact: 1 - {(L + 1) / 2:.1f} / {W} = {1 - (L + 1) / 2 / W:.1%})")
# 2. document mask cells
docs = [40, 50, 39]
print(f"2. document mask allows {sum(d * (d + 1) // 2 for d in docs):,} cells; causal mask {129 * 130 // 2:,}")
# 3. Jaccard, MinHash agreement
J = 300 / (400 + 360 - 300)
print(f"3. J = {J:.3f}; expected agreements {128 * J:.1f} of 128, spread {math.sqrt(128 * J * (1 - J)):.1f} positions"
      f" ({math.sqrt(J * (1 - J) / 128):.3f} in the estimate)")
# 4. banding
for b, r in ((32, 4), (16, 8), (64, 2)):
    p = lambda s: 1 - (1 - s ** r) ** b
    print(f"4. {b} bands of {r}: threshold {(1 / b) ** (1 / r):.3f}; chance at 0.3 {p(0.3):.3f}, at 0.8 {p(0.8):.4f}")
# 5. passes over a 25% source
pred = 384 * 8 * 0.25 * 128
print(f"5. {384 * 8 * 0.25:.0f} windows, {pred:,.0f} predictions over {44359:,} = {pred / 44359:.2f} passes")
# 6. shards
n = 2e12 / 2 ** 27
print(f"6. {math.ceil(n):,} shards of {2 ** 28:,} bytes; with 100,000 tokens uint32: {2 ** 29:,} bytes per shard")
# 7. buffer 1,000 over 10 sources of 100,000 windows, batches of 32
src = np.repeat(np.arange(10), 100_000)
buf, out = list(src[:1000]), []
pos = 1000
while pos + 32 <= len(src):
    batch = []
    for _ in range(32):
        k = rng.integers(len(buf)); batch.append(buf[k]); buf[k] = src[pos]; pos += 1
    out.append(np.bincount(batch, minlength=10).max() / 32)
out = np.array(out)
print(f"7. average largest share {out.mean():.3f}; batches with a single source {np.mean(out == 1):.1%}")
