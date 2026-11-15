"""Section 7.4: the same 1,024 x 1,024 agreement table three ways, timed.
Build machine: CPU only (8-core Intel Xeon, 16 GB), as of November 2026.
Run from 04-code/ch07:  python3 timing.py   (add --measure to time your own machine)"""
import _common
from _common import TINY_BPE, recorded
import time, os
import numpy as np
from tiny.corpus import load_text
from tiny.tokenizer import load_bpe, TINY_VOCAB_SIZE
from tiny.embed import Embedding, agreement_table, TINY_CONTEXT
from tiny.linalg import agreement_table_fast

tok = load_bpe(TINY_BPE)
W = Embedding(TINY_VOCAB_SIZE, seed=1865)(tok.encode(load_text())[:TINY_CONTEXT])   # (1024, 128)
ops = 2 * W.shape[0] * W.shape[0] * W.shape[1]

def best(f, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter(); f(); times.append(time.perf_counter() - t0)
    return min(times)

def measure():
    W32 = W.astype(np.float32)
    return {"loops": best(lambda: agreement_table(W), 1),                 # Chapter 6: 1,048,576 calls of dot
            "rows": best(lambda: np.array([w @ W.T for w in W]), 3),      # one row of the table at a time
            "line": best(lambda: agreement_table_fast(W), 20),            # W @ W.T
            "line32": best(lambda: W32 @ W32.T, 20)}                      # the same in 32-bit numbers
t = recorded("timing", measure)

print(f"window {W.shape}, table {agreement_table_fast(W).shape}, about {ops:,} operations")
for name, key in [("Chapter 6's loops", "loops"), ("one row at a time", "rows"),
                  ("one line, W @ W.T", "line"), ("one line, 32-bit numbers", "line32")]:
    print(f"{name:<26} {t[key] * 1000:>10.2f} ms   {ops / t[key] / 1e9:>8.2f} billion operations per second")
print(f"one line vs loops: {t['loops'] / t['line']:,.0f} times faster   (measured {t['measured']})")
small = W[:64]
print("the tables agree (checked on 64 rows):", bool(np.allclose(agreement_table(small), agreement_table_fast(small))))
