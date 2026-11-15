"""Chapter 7 Scoreboard.  Timings on the book's build machine (CPU only, 8-core Intel Xeon, 16 GB),
as of November 2026.  Run from 04-code/ch07:  python3 scoreboard.py   (--measure to retime)"""
import _common
from _common import TINY_BPE, recorded
import time
import numpy as np
from tiny.corpus import load_text
from tiny.tokenizer import load_bpe, TINY_VOCAB_SIZE
from tiny.embed import Embedding, PositionTable, TINY_D, TINY_CONTEXT
from tiny.linalg import agreement_table_fast

tok = load_bpe(TINY_BPE)
emb = Embedding(TINY_VOCAB_SIZE, TINY_D, seed=1865)
pos = PositionTable(TINY_CONTEXT, TINY_D, seed=1865)
block = emb(tok.encode(load_text())[:TINY_CONTEXT]) + pos(range(TINY_CONTEXT))     # (1024, 128)

def best(f, k=20):
    ts = []
    for _ in range(k):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts)
t = recorded("scoreboard", lambda: {"table": best(lambda: agreement_table_fast(block)),
                                     "scores": best(lambda: block @ emb.table.T)})
print(f"agreement table for one full window, {block.shape} @ {block.T.shape}: {t['table'] * 1000:.2f} ms "
      f"({TINY_CONTEXT / t['table']:,.0f} tokens per second)")
print(f"tied scores for one full window, {block.shape} @ {emb.table.T.shape}: {t['scores'] * 1000:.2f} ms "
      f"({TINY_CONTEXT / t['scores']:,.0f} tokens per second)   (measured {t['measured']})")
n = emb.table.size + pos.table.size
print(f"learned numbers: {emb.table.size:,} (tokens, tied) + {pos.table.size:,} (positions) = {n:,}; "
      f"tiny/linalg.py adds 0")
