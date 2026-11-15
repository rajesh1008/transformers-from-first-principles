"""Chapter 5 Scoreboard.  Timings on the book's build machine (CPU only), as of November 2026.
Run from 04-code/ch05:  python3 scoreboard.py"""
import _common, time
import numpy as np
from _common import TINY_BPE
from tiny.corpus import load_text
from tiny.tokenizer import load_bpe, TINY_VOCAB_SIZE
from tiny.embed import Embedding, PositionTable, sinusoidal, embed_with_positions, TINY_D, TINY_CONTEXT

tok = load_bpe(TINY_BPE)
ids = tok.encode(load_text())
emb = Embedding(TINY_VOCAB_SIZE, TINY_D, seed=1865)
pos = PositionTable(TINY_CONTEXT, TINY_D, seed=1865)
windows = [ids[i:i + TINY_CONTEXT] for i in range(0, len(ids), TINY_CONTEXT)]
t0 = time.perf_counter()
blocks = [embed_with_positions(w, emb, pos) for w in windows]
t = time.perf_counter() - t0
print(f"token row + position row for all {len(ids):,} tokens of the novel, in {len(windows)} windows: {t * 1000:.0f} ms "
      f"({len(ids) / t:,.0f} tokens per second)")
t0 = time.perf_counter(); S = sinusoidal(TINY_CONTEXT, TINY_D); t = time.perf_counter() - t0
print(f"computing all {TINY_CONTEXT:,} sinusoidal rows with loops: {t * 1000:.0f} ms")
n = emb.table.size + pos.table.size
print(f"learned numbers (weights) in Tiny so far: {emb.table.size:,} + {pos.table.size:,} = {n:,} ({n * 4 / 2**20:.1f} MiB)")
print("the running sentence is now", TINY_D, "numbers per token, token row plus position row; context", TINY_CONTEXT)
