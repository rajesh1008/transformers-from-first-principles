"""Chapter 12, section 6: tiny/ffn.py joins Tiny's layer.  Tiny's layer is now: add the four-head
attention output to each row, then add the feed-forward output to each row (normalisation waits for
Chapter 13).  Untrained, seed 1865: these numbers describe random tables, not learned memories.
Timings are the fastest of five, recorded once (--measure to time your machine).
Run from 04-code/ch12:  python3 tiny_ffn.py"""
from _common import *
import time
from tiny.attention import MultiHead
from tiny.ffn import FeedForward
from tiny.embed import Embedding, PositionTable
from tiny.tokenizer import load_bpe
from tiny.corpus import load_text

tok = load_bpe(TINY_BPE)
ids = tok.encode(SENTENCE)
pieces = [tok.decode([i]) for i in ids]
E, P = Embedding(len(tok)), PositionTable()
Xs = E(ids) + P(range(len(ids)))
text = load_text(); n = 1024
Xn = E(np.array(tok.encode(text))[:n]) + P(range(n))
att, ff = MultiHead(n_heads=4), FeedForward()

def layer(X):
    A = X + att(X)
    return A, A + ff(A)

A, Y = layer(Xs)
H = ff.hidden(A)
print(f"Tiny's layer on the running sentence ({len(ids)} tokens: {pieces})")
print(f"  feed-forward block: 128 -> {ff.d_ff} -> 128, GELU, {ff.n_params:,} learned numbers (untrained, seed 1865)")
for p, a, y, hrow in zip(pieces, A, Y, H):
    print(f"  {p!r:>8}: row length after attention {np.linalg.norm(a):.4f} | feed-forward adds a row of length "
          f"{np.linalg.norm(y - a):.5f} | neurons above zero {int((hrow > 0).sum())} of {ff.d_ff}")
alone = np.stack([ff(A[i:i + 1])[0] for i in range(len(A))])
print("  each row's feed-forward output computed alone equals the batched one:", bool(np.allclose(alone, ff(A))))

An, Yn = layer(Xn)
Hn = ff.hidden(An)
print(f"\non the first {n:,} tokens of the novel:")
print(f"  share of neuron activations above zero: {np.mean(Hn > 0):.3f}; largest activation {Hn.max():.4f}")
print(f"  mean row length after attention {np.linalg.norm(An, axis=1).mean():.4f}; mean length of the feed-forward addition "
      f"{np.linalg.norm(Yn - An, axis=1).mean():.5f}")
print(f"  ratio, feed-forward addition to row: {np.linalg.norm(Yn - An, axis=1).mean() / np.linalg.norm(An, axis=1).mean():.4f}"
      " (small because the tables are random with spread 0.02; training changes this)")
top = np.argsort(-Hn.max(0))[:3]
for j in top:
    k = int(Hn[:, j].argmax())
    print(f"  neuron {j:>3}: strongest on token {k} ({tok.decode([int(tok.encode(text)[k])])!r}), activation {Hn[k, j]:.4f}")

def measure():
    def best(f):
        ts = []
        for _ in range(5):
            t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
        return min(ts) * 1000
    return {"attention_ms": best(lambda: att(Xn)), "ffn_ms": best(lambda: ff(An)),
            "attention_tiled_ms": best(lambda: MultiHead(n_heads=4, block=64)(Xn)),
            "up_ms": best(lambda: An @ ff.W_1 + ff.b_1), "gelu_ms": best(lambda: gelu(U)), "relu_ms": best(lambda: relu(U)),
            "down_ms": best(lambda: G @ ff.W_2 + ff.b_2)}
from tiny.ffn import gelu, relu
U = An @ ff.W_1 + ff.b_1; G = gelu(U)
r = recorded("tiny_ffn_timing", measure)
print(f"\ntimings on {n:,} tokens, ms (fastest of 5, measured {r['measured']}):")
print(f"  four-head attention, plain {r['attention_ms']:.1f} | tiled block=64 {r['attention_tiled_ms']:.1f} | feed-forward block {r['ffn_ms']:.1f}")
print(f"  inside the feed-forward block: up table {r['up_ms']:.1f} | GELU {r['gelu_ms']:.1f} (ReLU would take {r['relu_ms']:.1f}) | down table {r['down_ms']:.1f}")
print(f"  operations: feed-forward 2 x 2 x {n} x 128 x 512 = {4 * n * 128 * 512 / 1e6:.1f}M; attention tables 4 x 2 x {n} x 128 x 128 = "
      f"{8 * n * 128 * 128 / 1e6:.1f}M plus scores and blend 2 x 2 x {n}^2 x 128 / 2 = {2 * n * n * 128 / 1e6:.1f}M")
g = FeedForward(gated=True, activation="silu")
print(f"\ngated variant (SiLU gate, d_ff {g.d_ff}, {g.n_params:,} numbers) on the sentence: share of hidden numbers above zero "
      f"{np.mean(g.hidden(A) > 0):.3f}; mean addition length {np.linalg.norm(g(A), axis=1).mean():.5f}")
