"""Chapter 13, sections 5 and 6: where to put the norm, measured on a stack of untrained Tiny layers.
Each layer is Tiny's four-head attention and its feed-forward block, with its own seed (1865 + 100 l).
Five wirings (see tiny/norm.py layer_step; 'keel' is the January 2026 paper's x <- LN(a x + F(LN(x)))
with a = the number of residual connections in the stack).  Input: Tiny's rows for the first 256 tokens
of the novel.  Two spreads for the tables: Tiny's 0.02, and 1/sqrt(128) = 0.088, at which a table keeps
a row's length on average (closer to the size trained blocks reach).  Untrained: this measures how the
wiring treats the numbers, not what a trained model does with them.
Measured at depths 1, 2, 4, 6, 12, 24, 48:
  length   mean row length at the top of the stack
  carried  the length of the unchanged copy of the input row inside the top row, as a share of the top
           row's length (exact bookkeeping: pre-norm and residual keep x0 itself; post-norm and keel keep
           a rescaled copy of x0's centred row, rescaled by every norm it passes through)
  similar  token similarity of the top table (1 = every row the same)
Run from 04-code/ch13:  python3 placement.py   (about a minute)"""
from _common import *
from tiny.attention import MultiHead
from tiny.ffn import FeedForward
from tiny.norm import LayerNorm, layer_step, token_similarity, layer_norm

tok, ids, X0 = tiny_rows(256)
DEPTHS = [1, 2, 4, 6, 12, 24, 48]
D = 128
norm = LayerNorm(D)

def stack(n_layers, spread):
    return [(MultiHead(n_heads=4, seed=SEED + 100 * l, spread=spread), FeedForward(seed=SEED + 100 * l, spread=spread))
            for l in range(n_layers)]

def run(layers, wiring, x0, depths):
    x = x0.copy(); out = {}
    c = np.ones((len(x0), 1))                          # coefficient of x0's carried copy (post, keel)
    x0c = x0 - x0.mean(-1, keepdims=True)
    a = 2.0 * len(layers)
    k = 0
    for l, (att, ff) in enumerate(layers):
        for f in (att, ff):
            k += 1
            if wiring == "keel":
                z = a * x + f(norm(x))
                s = np.sqrt(z.var(-1, keepdims=True) + 1e-5)
                x = layer_norm(z); c = c * a / s
            elif wiring == "post":
                z = x + f(x)
                s = np.sqrt(z.var(-1, keepdims=True) + 1e-5)
                x = layer_norm(z); c = c / s
            else:
                x = layer_step(x, f, norm, wiring)
        if l + 1 in depths:
            L = np.linalg.norm(x, axis=1)
            if wiring in ("pre", "residual"):
                carried = np.linalg.norm(x0, axis=1) / L
            elif wiring in ("post", "keel"):
                carried = (c[:, 0] * np.linalg.norm(x0c, axis=1)) / L
            else:
                carried = np.zeros(len(x))
            out[l + 1] = (float(L.mean()), float(carried.mean()), token_similarity(x))
    return out

results = {}
for spread in (0.02, 1 / np.sqrt(D)):
    layers = stack(max(DEPTHS), spread)
    att, ff = layers[0]
    Xn = layer_norm(X0)
    print(f"\ntable spread {spread:.3f}: one attention output / its input, on normalised rows: "
          f"{np.linalg.norm(att(Xn), axis=1).mean() / np.linalg.norm(Xn, axis=1).mean():.3f}; "
          f"one feed-forward output / input: {np.linalg.norm(ff(Xn), axis=1).mean() / np.linalg.norm(Xn, axis=1).mean():.3f}")
    print(f"  input rows: mean length {np.linalg.norm(X0, axis=1).mean():.4f}, token similarity {token_similarity(X0):.3f}")
    print(f"  {'wiring':<9}" + "".join(f"| depth {d:<2}                 " for d in DEPTHS[:1]) + " (length / carried / similar)")
    for wiring in ("none", "residual", "post", "pre", "keel"):
        r = run(layers, wiring, X0, DEPTHS)
        results[f"{spread:.3f}/{wiring}"] = r
        print(f"  {wiring:<9}" + " ".join(f"{d}: {r[d][0]:.3g}/{r[d][1]:.3f}/{r[d][2]:.3f}" for d in DEPTHS))
json.dump(results, open(os.path.join(HERE, "outputs", "placement.json"), "w"), indent=1)

# the 1/sqrt(2) rule for post-norm: when a block's output is the same size as its input and unrelated
rng = np.random.default_rng(SEED)
x = rng.normal(size=(1, D)); x = layer_norm(x); c = 1.0
for k in range(1, 13):
    f = rng.normal(size=(1, D)); f = f / np.linalg.norm(f) * np.linalg.norm(x)
    z = x + f; s = np.sqrt(z.var() + 1e-5); x = layer_norm(z); c = c / s
    if k in (1, 2, 4, 8, 12):
        print(f"post-norm, block output the same size as the row, unrelated: after {k:>2} additions the carried copy is "
              f"{c:.4f} of the original (1/sqrt(2)^{k} = {2 ** (-k / 2):.4f})")
