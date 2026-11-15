"""Tiny's embedding table: 2,048 rows of 128 numbers, and how the 2-D toy table maps onto it.
Run from 04-code/ch04:  python3 tiny_table.py"""
import _common
from _common import SENTENCE, TINY_BPE
import numpy as np
from collections import Counter
from tiny.corpus import load_text
from tiny.tokenizer import load_bpe, TINY_VOCAB_SIZE, TOY_VOCAB
from tiny.embed import Embedding, TINY_D, TOY_EMBED

tok = load_bpe(TINY_BPE)
emb = Embedding(TINY_VOCAB_SIZE, TINY_D, seed=1865)
print(f"Tiny's table: {emb.shape[0]:,} rows x {emb.shape[1]} numbers = {emb.table.size:,} numbers "
      f"= {emb.table.size * 4:,} bytes at 4 bytes each")
ids = tok.encode(SENTENCE)
X = emb(ids)
print(f"{SENTENCE!r} -> Tiny ids {ids} -> {X.shape[0]} rows of {X.shape[1]} numbers")
print("toy row (2 numbers, for hand work)  ->  Tiny row (128 numbers, for code)")
for i, piece in zip(ids, [tok.piece(i) for i in ids]):
    word = piece.replace("\u2423", "")
    toy = "(half of mat)" if i in (283, 303) else TOY_EMBED[TOY_VOCAB.index(word)].astype(int).tolist()
    print(f"  {piece:<5} toy {str(toy):<14} Tiny id {i:>4}: first 4 = {np.round(emb.table[i][:4], 4).tolist()}")
r = emb.table[725]
print(f"length of the row for ' cat' (id 725): {np.linalg.norm(r):.3f}; "
      f"expected about 0.02 x sqrt(128) = {0.02 * 128 ** 0.5:.3f}")

used = Counter(tok.encode(load_text()))
print(f"\nrows the novel will ever train: {len(used):,} of {TINY_VOCAB_SIZE:,}; "
      f"never used by the novel: {TINY_VOCAB_SIZE - len(used):,}")

rng = np.random.default_rng(1865)
for d in (2, 16, 128, 768):
    a, b = rng.normal(size=(5000, d)), rng.normal(size=(5000, d))
    c = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    print(f"random arrows in {d:>3} dimensions: average size of cosine {np.abs(c).mean():.3f}; "
          f"within 10 degrees of a right angle {np.mean(np.abs(c) < np.sin(np.radians(10))) * 100:.0f}%")
