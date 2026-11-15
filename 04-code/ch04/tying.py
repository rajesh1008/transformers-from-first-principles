"""Tying input and output embeddings: one table to read tokens in and to score tokens out.
Run from 04-code/ch04:  python3 tying.py"""
import _common, _gpt2
import numpy as np
from tiny.tokenizer import TOY_VOCAB
from tiny.embed import TOY_EMBED, dot, TINY_D, Embedding

# Reading in, as dot products: a list of zeros with a single 1 picks out one row
one_hot = [1 if i == 1 else 0 for i in range(len(TOY_VOCAB))]          # id 1 = cat
cols = [TOY_EMBED[:, 0], TOY_EMBED[:, 1]]
print("one-hot list for id 1:", one_hot)
print("its dot product with each column of the table:", [int(dot(one_hot, c)) for c in cols],
      "= cat's row", TOY_EMBED[1].astype(int).tolist())

# Scoring out: the dot product of the model's final vector with every row
h = (-1.2, 1.0)
scores = [dot(h, row) for row in TOY_EMBED]
print(f"\nfinal vector h = {h}; its agreement with every row of the same table:")
for w, s in sorted(zip(TOY_VOCAB, scores), key=lambda t: -t[1]):
    print(f"  {w:<7} {s:>5.1f}")
e = Embedding.__new__(Embedding); e.table = TOY_EMBED
print("tiny.embed scores() gives the same list:", np.round(e.scores(np.array(h)), 1).tolist())

# GPT-2 stores one table and uses it at both ends
hdr, _ = _gpt2.header()
names = [k for k in hdr if k != "__metadata__"]
total = sum(int(np.prod(hdr[k]["shape"])) for k in names if not k.endswith(".attn.bias"))   # .attn.bias is a fixed mask, not learned
print(f"\nGPT-2's file: {len(names)} tensors; any separate output table? "
      f"{any('lm_head' in k for k in names)}; token table 'wte.weight' {hdr['wte.weight']['shape']}")
V, D = hdr["wte.weight"]["shape"]
print(f"GPT-2 learned numbers: {total:,}; token table {V * D:,} ({V * D / total * 100:.0f}%); "
      f"untied, it would be {total + V * D:,}")
print(f"Tiny: tying saves 2,048 x {TINY_D} = {2048 * TINY_D:,} numbers")
