"""Chapter 22, s22-8: the yardstick for Part 4 of stream.py.  If every window of a batch were drawn from the whole
pool at random (a full shuffle), what would the largest share of one book in a batch of 8 be, on average?
Simulated with 50,000 batches, seed 1865, books weighted by their window counts."""
import _common
import numpy as np
from _pool import pool_windows, POOL_BOOKS
from _common import SEED, save_json
_, B = pool_windows()
w = np.bincount(B[:, 0], minlength=len(POOL_BOOKS)).astype(float)
x = np.random.default_rng(SEED).choice(len(w), size=(50000, 8), p=w / w.sum())
share = np.mean([np.bincount(r, minlength=len(w)).max() for r in x]) / 8
print(f"windows per book: {', '.join(f'{b} {int(n)}' for b, n in zip(POOL_BOOKS, w))}")
print(f"full shuffle: largest share of one book in a batch of 8, average {share:.2f}")
save_json("full_shuffle", dict(share=float(share)))
