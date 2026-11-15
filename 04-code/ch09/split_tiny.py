"""Chapter 9: splitting Tiny's width.  One head of width 128 (Chapter 8) and four heads of width 32
use the *same* query, key and value numbers (same seeds), cut differently.  The single head's raw
score table is the sum of the four heads' raw score tables; what changes is that each head of 32
is scaled and softmaxed on its own.  Learned numbers: 3 x 128 x 128 = 49,152 either way, plus the
output table, 128 x 128 = 16,384.  Run from 04-code/ch09:  python3 split_tiny.py"""
from _common import *
from tiny.attention import Head, MultiHead, entropy, split_heads
from tiny.embed import Embedding, PositionTable
from tiny.tokenizer import load_bpe

tok = load_bpe(TINY_BPE)
ids = tok.encode(SENTENCE)
pieces = [tok.piece(i) for i in ids]
emb, pos = Embedding(len(tok)), PositionTable()
B = emb(ids) + pos(range(len(ids)))
one = Head()                                   # Chapter 8's head: d_head = 128
four = MultiHead(n_heads=4)                    # Chapter 9: four heads of 32
print("Tiny's pieces:", pieces, " rows", B.shape)
print("same query numbers as Chapter 8's head:", bool(np.allclose(one.W_q, four.W_q)),
      "| same keys:", bool(np.allclose(one.W_k, four.W_k)), "| same values:", bool(np.allclose(one.W_v, four.W_v)))
print(f"learned numbers: one head {one.n_params:,}; four heads {3 * 128 * 128:,} + output table {four.W_o.size:,} = {four.n_params:,}")
Q, K = B @ four.W_q, B @ four.W_k
raw4 = split_heads(Q, 4) @ split_heads(K, 4).transpose(0, 2, 1)
print("one head's raw table equals the sum of the four heads' raw tables:", bool(np.allclose(raw4.sum(0), Q @ K.T)))
_, A1 = one(B, return_shares=True)
outs, A4 = four.head_outputs(B)
print("\nlargest raw score: one head", f"{np.abs(Q @ K.T).max():.2e}", " four heads", [f"{np.abs(r).max():.2e}" for r in raw4])
print("entropy of the last row (7 seats; even split = ln 7 = 1.946):")
print("  one head of 128:", f"{entropy(A1)[-1]:.4f}", "  four heads of 32:", [f"{e:.4f}" for e in entropy(A4)[:, -1]])
print("an untrained head is an averager whatever its width: last row of head 0 ->", r3(A4[0, -1]))
out = four(B)
print("output rows", out.shape, " sum", f"{out.sum():.6f}")
