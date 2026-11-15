"""Section 7.5: batches.  Three sentences stacked into one block of shape (3, 6, 2); then the cost of
scoring B tokens at once against Tiny's 2,048-row table.  Build machine: one CPU core, as of November 2026.
Run from 04-code/ch07:  python3 batches.py   (add --measure to time your own machine)"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"                         # one core, so the curve is about batching alone
import _common
from _common import TINY_BPE, recorded
import time
import numpy as np
from tiny.tokenizer import TOY_VOCAB, TINY_VOCAB_SIZE
from tiny.embed import TOY_EMBED, Embedding

def rows(text, length=6):
    ids = [TOY_VOCAB.index(w) for w in text.split()]
    ids += [TOY_VOCAB.index("<unk>")] * (length - len(ids))      # pad short sentences with (0, 0) rows
    return TOY_EMBED[ids]

batch = np.stack([rows("the cat sat on the mat"), rows("the mat sat on the cat"), rows("the rabbit looked at")])
tables = batch @ batch.transpose(0, 2, 1)                         # (3, 6, 2) @ (3, 2, 6) -> (3, 6, 6)
print("batch", batch.shape, "@ batch turned", batch.transpose(0, 2, 1).shape, "->", tables.shape)
print("sentence 0, row the:", tables[0, 0].tolist())
print("sentence 2 (the rabbit looked at + 2 padding rows), row rabbit:", tables[2, 1].tolist())
print("each table equals its own X @ X.T:", all(np.array_equal(tables[i], b @ b.T) for i, b in enumerate(batch)))

E = Embedding(TINY_VOCAB_SIZE, seed=1865).table                  # (2048, 128), Tiny's tied table
SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
def measure():
    rng = np.random.default_rng(1865); out = {}
    for b in SIZES:
        H = rng.normal(0, 0.02, size=(b, 128))
        reps = max(5, 4000 // b); best = 1e9
        for _ in range(9):
            t0 = time.perf_counter()
            for _ in range(reps): H @ E.T                          # (b, 128) @ (128, 2048): b rows of 2,048 scores
            best = min(best, (time.perf_counter() - t0) / reps)
        out[str(b)] = best
    return out
t = recorded("batches", measure)
print(f"\n{'rows at once':>12}{'one call (ms)':>15}{'per row (ms)':>14}{'rows per second':>17}")
for b in SIZES:
    s = t[str(b)]
    print(f"{b:>12}{s * 1000:>15.3f}{s / b * 1000:>14.4f}{b / s:>17,.0f}")
n = np.array(SIZES, float); y = np.array([t[str(b)] for b in SIZES])
beta, alpha = np.polyfit(n, y, 1)                                 # one call = alpha + beta * rows
print(f"fit: one call = {alpha * 1e6:.1f} microseconds + {beta * 1e6:.2f} microseconds per row  (measured {t['measured']})")
print(f"per row at 1 and 1024 rows at once: {t['1'] / t[str(1024)] * 1024:,.0f} times cheaper batched")
