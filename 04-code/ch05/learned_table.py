"""Section 5.3: Tiny's learned position table, before training.  Run: python3 learned_table.py"""
import _common
from _common import SENTENCE, TINY_BPE
import numpy as np
from tiny.tokenizer import load_bpe, TINY_VOCAB_SIZE
from tiny.embed import Embedding, PositionTable, embed_with_positions, TINY_CONTEXT, TINY_D, cosine, length

tok = load_bpe(TINY_BPE)
ids = tok.encode(SENTENCE)
emb = Embedding(TINY_VOCAB_SIZE, seed=1865)
pos = PositionTable(TINY_CONTEXT, TINY_D, seed=1865)
print("position table:", pos.shape, f"= {pos.table.size:,} learned numbers")
print("row 0 starts", np.round(pos.table[0][:4], 4).tolist())
print("ids", ids, "-> positions", list(range(len(ids))))
X = embed_with_positions(ids, emb, pos)
print("input block", X.shape, "first row starts", np.round(X[0][:4], 4).tolist(),
      "= token row", np.round(emb.table[ids[0]][:4], 4).tolist(), "+ position row", np.round(pos.table[0][:4], 4).tolist())
print(f"typical lengths before training: token row {np.median([length(r) for r in emb.table]):.3f}, "
      f"position row {np.median([length(r) for r in pos.table]):.3f}")
print("untrained: cosine of position rows 0 and 1:", round(cosine(pos.table[0], pos.table[1]), 3),
      "; rows 0 and 500:", round(cosine(pos.table[0], pos.table[500]), 3))
print(f"learned numbers in Tiny so far: {emb.table.size:,} (tokens) + {pos.table.size:,} (positions) = "
      f"{emb.table.size + pos.table.size:,}")
