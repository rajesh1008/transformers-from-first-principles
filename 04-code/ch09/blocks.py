"""Chapter 9 Interlude: cutting a table into blocks.  Putting the heads' outputs side by side and
multiplying by the output table gives exactly the same rows as multiplying each head's output by
its own block of rows of the output table and adding the results.  Run: python3 blocks.py"""
from _common import *
a = np.array([[1.0, 2.0]])                 # head 1's output for one token (width 2)
b = np.array([[3.0, -1.0]])                # head 2's output for the same token
W_o = np.array([[1, 0, 2], [0, 1, 1], [1, 1, 0], [2, 0, -1]], dtype=float)   # (2 + 2) x 3
side = np.concatenate([a, b], axis=1)
print("side by side:", side.tolist())
print("side by side @ W_o:", (side @ W_o).tolist())
print("head 1 @ rows 1-2 of W_o:", (a @ W_o[:2]).tolist(), "  head 2 @ rows 3-4 of W_o:", (b @ W_o[2:]).tolist())
print("their sum:", (a @ W_o[:2] + b @ W_o[2:]).tolist())
# the same with Tiny-sized random numbers: four heads of 32 into 128
rng = np.random.default_rng(SEED)
outs = [rng.normal(size=(7, 32)) for _ in range(4)]
Wo = rng.normal(size=(128, 128))
joined = np.concatenate(outs, axis=1) @ Wo
summed = sum(o @ Wo[32 * h:32 * (h + 1)] for h, o in enumerate(outs))
print(f"Tiny-sized check (seed {SEED}): largest difference between the two ways: {np.abs(joined - summed).max():.1e}")
# Check yourself answers
print("check 1:", (np.array([[1.0, 0.0, 2.0, 1.0]]) @ np.array([[1, 1], [0, 2], [1, 0], [3, 1]], dtype=float)).tolist())
print("check 2 (head 2 removed):", (np.array([[1.0, 0.0]]) @ np.array([[1, 1], [0, 2]], dtype=float)).tolist())
