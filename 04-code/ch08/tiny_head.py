"""Chapter 8: Tiny's first head on the running sentence (Tiny's tokenizer, width 128, learned
positions), and the same head on a full 1,024-token window of the novel.
Run from 04-code/ch08:  python3 tiny_head.py   (add --measure to time your own machine)"""
from _common import *
import time
from tiny.attention import Head, entropy
from tiny.embed import Embedding, PositionTable
from tiny.tokenizer import load_bpe
from tiny.corpus import load_text

tok = load_bpe(TINY_BPE)
ids = tok.encode(SENTENCE)
emb, pos = Embedding(len(tok)), PositionTable()
B = emb(ids) + pos(range(len(ids)))                     # (7, 128)
head = Head()
out, A = head(B, return_shares=True)
pieces = [tok.piece(i) for i in ids]
print("ids", ids, "pieces", pieces, "block", B.shape)
print("head tables 3 x (128 x 128) =", f"{head.n_params:,}", "learned numbers")
print("scores before softmax, largest size:", f"{np.abs((B @ head.W_q) @ (B @ head.W_k).T / np.sqrt(128)).max():.2e}")
print("shares (rows ask, columns answer):")
print(table(A, rows=pieces, cols=pieces, fmt="{:>7.3f}"))
print("output block", out.shape, "sum", f"{out.sum():.6f}")
print("entropy of each row vs an even split:", [f"{e:.3f}/{np.log(k + 1):.3f}" for k, e in enumerate(entropy(A))])

novel = np.array(tok.encode(load_text()))[:1024]
W = emb(novel) + pos(range(1024))
def measure():
    t = time.perf_counter(); runs = 20
    for _ in range(runs):
        head(W)
    return {"ms": (time.perf_counter() - t) * 1000 / runs}
r = recorded("tiny_head_window", measure) if "recorded" in globals() else measure()
print(f"one head on a full window (1,024 x 128): {r['ms']:.2f} ms")
