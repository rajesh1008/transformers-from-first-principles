"""Chapter 9, section 13: using the extended tiny/attention.py on Tiny's rows for the running sentence.
Run from 04-code/ch09:  python3 usage.py"""
from _common import *
from tiny.embed import Embedding, PositionTable
from tiny.tokenizer import load_bpe
tok = load_bpe(TINY_BPE); ids = tok.encode(SENTENCE)
X = Embedding(len(tok))(ids) + PositionTable()(range(len(ids)))   # Tiny's rows: (7, 128)

from tiny.attention import Head, MultiHead
h = Head(d_head=32, rotary=True)       # one narrow head with rotary positions: 12,288 numbers
m = MultiHead(n_heads=4)               # four heads of 32 and an output table: 65,536 numbers
out, shares = m(X, return_shares=True) # X (n, 128) -> out (n, 128), shares (4, n, n)
out_minus_2 = m(X, ablate=[2])         # head 2 removed, correctly, before the output table

print("numbers:", f"{h.n_params:,}", f"{m.n_params:,}", "| shapes:", X.shape, out.shape, shares.shape)
outs, _ = m.head_outputs(X)
print("removing head 2 takes away exactly its own contribution:",
      bool(np.allclose(out - out_minus_2, outs[2] @ m.W_o[64:96])))
