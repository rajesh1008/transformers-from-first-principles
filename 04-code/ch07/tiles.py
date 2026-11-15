"""Section 7.4: computing the window's table one tile at a time.  Same answer; far fewer trips to memory.
Run from 04-code/ch07:  python3 tiles.py"""
import _common
from _common import TINY_BPE
import numpy as np
from tiny.corpus import load_text
from tiny.tokenizer import load_bpe, TINY_VOCAB_SIZE
from tiny.embed import Embedding, TINY_CONTEXT
from tiny.linalg import matmul_tiled

tok = load_bpe(TINY_BPE)
W = Embedding(TINY_VOCAB_SIZE, seed=1865)(tok.encode(load_text())[:TINY_CONTEXT])   # (1024, 128)
T = W @ W.T
for tile in [64, 128, 256]:
    print(f"tile {tile:>3} x {tile:<3}: same table as W @ W.T: {bool(np.allclose(matmul_tiled(W, W.T, tile), T))}")

n, d = W.shape
print(f"\nnumbers read from far memory to build the {n} x {n} table (inputs only):")
for tile in [1, 8, 32, 128, 512, 1024]:
    tiles = (n // tile) ** 2
    reads = tiles * 2 * tile * d               # each answer tile needs a strip of `tile` rows from each side
    print(f"  tiles of {tile:>4} x {tile:<4}: {tiles:>9,} tiles, {reads:>12,} numbers read "
          f"({reads / (2 * n * d):>6,.0f} times the {2 * n * d:,} numbers that exist)")
print(f"\nthe table itself: {n * n:,} numbers = {n * n * 2 / 2**20:.0f} MiB in 16-bit, for one head of one layer")
