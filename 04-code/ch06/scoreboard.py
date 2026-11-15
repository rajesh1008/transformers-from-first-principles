"""Chapter 6 Scoreboard.  Timings on the book's build machine (CPU only), as of November 2026.
Run from 04-code/ch06:  python3 scoreboard.py"""
import _common, time
import numpy as np
from _common import TINY_BPE
from tiny.corpus import load_text
from tiny.tokenizer import load_bpe, TINY_VOCAB_SIZE
from tiny.embed import Embedding, PositionTable, Rotary, TINY_D, TINY_CONTEXT

tok = load_bpe(TINY_BPE)
ids = tok.encode(load_text())
emb = Embedding(TINY_VOCAB_SIZE, TINY_D, seed=1865)
pos = PositionTable(TINY_CONTEXT, TINY_D, seed=1865)
rot = Rotary(TINY_D)
window = emb(ids[:TINY_CONTEXT])
t0 = time.perf_counter(); R = rot(window); t = time.perf_counter() - t0
print(f"rotary on one full window ({TINY_CONTEXT:,} x {TINY_D}) with plain loops: {t * 1000:.0f} ms "
      f"({TINY_CONTEXT / t:,.0f} tokens per second)")
print(f"angles turned per window: {TINY_CONTEXT:,} tokens x {len(rot.speeds)} clocks = {TINY_CONTEXT * len(rot.speeds):,}")
n = emb.table.size + pos.table.size
print(f"learned numbers: {emb.table.size:,} (tokens, tied) + {pos.table.size:,} (positions, default) = {n:,}; "
      f"rotary option adds 0")
