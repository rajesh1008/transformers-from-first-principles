"""Chapter 11, section 8: tiny/attention.py gains tiled_attention and MultiHead(block=...).  Checks that the
tiled path gives Chapter 8's head, Chapter 9's four heads and Chapter 10's variants to rounding, on the running
sentence and on the first 1,024 tokens of the novel, and times Tiny's layer both ways (fastest of five,
recorded once; --measure to time your machine).  Run from 04-code/ch11:  python3 tiny_tiled.py"""
from _common import *
import time
from tiny.attention import attention, tiled_attention, MultiHead, Head, window_mask, softmax
from tiny.embed import Embedding, PositionTable
from tiny.tokenizer import load_bpe
from tiny.corpus import load_text

# 1. Chapter 8's hand head on the running sentence, in blocks of 3 (two tiles each way)
out8, _ = attention(X, W_q, W_k, W_v)
t8 = tiled_attention(X @ W_q, X @ W_k, X @ W_v, block=3)
print("hand head, blocks of 3: equal to attention():", bool(np.allclose(out8, t8)),
      f"| largest difference {np.abs(out8 - t8).max():.1e}")
print("  mat's output, tiled:", r3(t8[5]), " (Chapter 8: [0.665, -0.297] without positions)")

tok = load_bpe(TINY_BPE)
ids = tok.encode(SENTENCE)
Xs = Embedding(len(tok))(ids) + PositionTable()(range(len(ids)))
text = load_text()
all_ids = np.array(tok.encode(text))
n = 1024
Xn = Embedding(len(tok))(all_ids[:n]) + PositionTable()(range(n))

cases = [("Chapter 8: one head of 128", dict(n_heads=1, d_head=128)),
         ("Chapter 9: four heads of 32", dict(n_heads=4)),
         ("Chapter 10: 2 kv heads", dict(n_heads=4, n_kv_heads=2)),
         ("Chapter 10: window 256", dict(n_heads=4, window=256)),
         ("Chapter 10: 2 kv heads + window 256", dict(n_heads=4, n_kv_heads=2, window=256))]
print(f"\nTiny's layer, plain against block=64 (running sentence: {len(ids)} tokens; novel: first {n:,}):")
worst = 0.0
for name, kw in cases:
    a, b = MultiHead(**kw)(Xs), MultiHead(block=64, **kw)(Xs)
    c, e = MultiHead(**kw)(Xn), MultiHead(block=64, **kw)(Xn)
    d1, d2 = np.abs(a - b).max(), np.abs(c - e).max()
    worst = max(worst, d1, d2)
    print(f"  {name:<36} sentence {d1:.1e} | novel {d2:.1e} | equal {bool(np.allclose(a, b) and np.allclose(c, e))}")
print(f"largest difference anywhere: {worst:.1e}")
scale = np.abs(MultiHead(n_heads=4)(Xn)).max()
print(f"largest output number (four heads, novel): {scale:.1e}, so the largest difference is {worst / scale:.1e} of it")
h = Head()
print("Head() and tiled_attention on Head's own tables agree on the novel:",
      bool(np.allclose(h(Xn), (lambda m: (np.stack([tiled_attention(Xn @ m.W_q, Xn @ m.W_k, Xn @ m.W_v, block=64)])[0]))(h))))

def measure():
    res = {}
    for name, kw in [("multi-head", dict(n_heads=4)), ("window 256", dict(n_heads=4, window=256))]:
        for label, extra in [("plain", {}), ("block 64", dict(block=64)), ("block 256", dict(block=256))]:
            m = MultiHead(**kw, **extra)
            best = 1e9
            for _ in range(5):
                t = time.perf_counter(); m(Xn); best = min(best, time.perf_counter() - t)
            res[f"{name} {label}"] = round(best * 1000, 1)
    return res
times = recorded("tiny_tiled_timing", measure)
print(f"\nTiny's layer on the first 1,024 tokens, ms (fastest of 5, measured {times['measured']}):")
for name in ("multi-head", "window 256"):
    print(f"  {name:<11} plain {times[name + ' plain']:>6} | block 64 {times[name + ' block 64']:>6} | block 256 {times[name + ' block 256']:>6}")
