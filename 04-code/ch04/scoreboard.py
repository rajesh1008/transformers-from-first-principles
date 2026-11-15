"""Chapter 4 Scoreboard.  Timings on the book's build machine (CPU only), as of November 2026.
Run from 04-code/ch04:  python3 scoreboard.py"""
import _common, time
import numpy as np
from _common import TINY_BPE
from tiny.corpus import load_text
from tiny.tokenizer import load_bpe, TINY_VOCAB_SIZE
from tiny.embed import Embedding, TINY_D
from nudge import train

tok = load_bpe(TINY_BPE)
ids = np.array(tok.encode(load_text()))
emb = Embedding(TINY_VOCAB_SIZE, TINY_D, seed=1865)
t0 = time.perf_counter(); X = emb(ids); t = time.perf_counter() - t0
print(f"looking up all {len(ids):,} tokens of the novel in Tiny's table: {t * 1000:.0f} ms -> block {X.shape}")
t0 = time.perf_counter(); train(d=16, epochs=10, seed=1865); t = time.perf_counter() - t0
print(f"learning 16-number rows for 699 words, 10 passes: {t:.1f} s")
print(f"learned numbers (weights) in Tiny so far: {emb.table.size:,} (the token table, {emb.table.size * 4 / 2**20:.0f} MiB)")
