"""Chapter 13, section 9: tiny/norm.py joins Tiny's layer.  Tiny's layer becomes pre-norm, as GPT-2's:
x <- x + MultiHead(ln_1(x)); x <- x + FeedForward(ln_2(x)).  Untrained, seed 1865, gains one, biases zero.
Timings on 1,024 tokens of the novel are the fastest of five, recorded once (--measure to time yours).
Run from 04-code/ch13:  python3 tiny_norm.py"""
from _common import *
import time
from tiny.attention import MultiHead
from tiny.ffn import FeedForward
from tiny.norm import LayerNorm, RMSNorm, token_similarity

tok, ids, Xs = tiny_rows()
pieces = [tok.decode([i]) for i in ids]
att, ff = MultiHead(n_heads=4), FeedForward()
ln_1, ln_2 = LayerNorm(128), LayerNorm(128)

def layer_ch12(X):
    A = X + att(X); return A + ff(A)

def layer_ch13(X):
    A = X + att(ln_1(X)); return A, A + ff(ln_2(A))

A, Y = layer_ch13(Xs)
Y12 = layer_ch12(Xs)
print(f"Tiny's layer, pre-norm, on the running sentence ({len(ids)} tokens: {pieces})")
print(f"  learned numbers: two layer norms of 128 gains and 128 biases = {ln_1.n_params + ln_2.n_params}; "
      f"layer total {att.n_params + ff.n_params + ln_1.n_params + ln_2.n_params:,}; Tiny {590464 + ln_1.n_params + ln_2.n_params:,}")
print("  token | row in | attention adds (Ch12 / Ch13) | feed-forward adds (Ch12 / Ch13) | row out")
A12 = Xs + att(Xs)
for p, x, a12, y12, a, y in zip(pieces, Xs, A12, Y12, A, Y):
    print(f"  {p!r:>7} | {np.linalg.norm(x):.4f} | {np.linalg.norm(a12 - x):.5f} / {np.linalg.norm(a - x):.4f} | "
          f"{np.linalg.norm(y12 - a12):.5f} / {np.linalg.norm(y - a):.4f} | {np.linalg.norm(y):.4f}")
k = pieces.index(" sat")
z = ln_1(Xs)[k]
print(f"\nthe checkpoint row: ' sat' (token {k}) entering the layer, before and after ln_1")
print(f"  before: average {Xs[k].mean():.5f}, spread {Xs[k].std():.5f}, length {np.linalg.norm(Xs[k]):.4f}, first four {r3(Xs[k][:4], 4)}")
print(f"  after:  average {z.mean():.6f}, spread {z.std():.4f}, length {np.linalg.norm(z):.3f}, first four {r3(z[:4], 3)}")
print(f"  largest number after: {z.max():.3f} at column {int(z.argmax())}; smallest {z.min():.3f} at column {int(z.argmin())}")
print("  each row normalised alone equals the batched result:", bool(np.allclose(np.stack([ln_1(Xs[i:i + 1])[0] for i in range(len(Xs))]), ln_1(Xs))))

_, _, Xn = tiny_rows(1024)
An, Yn = layer_ch13(Xn)
print(f"\non the first 1,024 tokens of the novel: mean row length in {np.linalg.norm(Xn, axis=1).mean():.4f}, "
      f"after attention {np.linalg.norm(An, axis=1).mean():.4f}, out {np.linalg.norm(Yn, axis=1).mean():.4f}")
print(f"  token similarity in {token_similarity(Xn):.3f}, out {token_similarity(Yn):.3f}")
print(f"  the blocks now see rows of length about {np.linalg.norm(ln_1(Xn), axis=1).mean():.2f}; "
      f"their outputs are {np.linalg.norm(An - Xn, axis=1).mean():.4f} and {np.linalg.norm(Yn - An, axis=1).mean():.4f} long")

rn = RMSNorm(128)
def measure():
    def best(f):
        ts = []
        for _ in range(5):
            t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
        return min(ts) * 1000
    return {"layer_norm_ms": best(lambda: ln_1(Xn)), "rms_norm_ms": best(lambda: rn(Xn)),
            "attention_ms": best(lambda: att(ln_1(Xn))), "ffn_ms": best(lambda: ff(ln_2(An))),
            "layer_ms": best(lambda: layer_ch13(Xn))}
t = recorded("tiny_norm_timing", measure)
print(f"\ntimings on 1,024 tokens, ms (fastest of 5, measured {t['measured']}):")
print(f"  layer norm {t['layer_norm_ms']:.2f} | RMSNorm {t['rms_norm_ms']:.2f} | attention {t['attention_ms']:.1f} | "
      f"feed-forward {t['ffn_ms']:.1f} | whole layer {t['layer_ms']:.1f}")
