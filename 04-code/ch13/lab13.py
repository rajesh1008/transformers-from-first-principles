"""Lab 13: normalise sat's row by hand and in Tiny, then three breakages that run without an error,
and the checkpoint tests that catch them.  Breakage 1 removes the residual additions from a stack of
twelve untrained Tiny layers (float32, as most training code uses); Breakage 2 normalises over the
tokens instead of over each row; Breakage 3 uses the d - 1 ('sample') spread.
Run from 04-code/ch13:  python3 lab13.py"""
from _common import *
from tiny.attention import MultiHead
from tiny.ffn import FeedForward
from tiny.norm import LayerNorm, layer_norm, rms_norm, token_similarity

# Part 1
A, Y = hand_layer(X)
sat = Y[2]
print("Part 1. sat's row after Chapter 12's hand layer, normalised")
print(f"  sat {r3(sat)}: average {sat.mean():.3f}, spread {sat.std():.3f} -> layer norm {r3(layer_norm(sat), 5)} | RMSNorm {r3(rms_norm(sat))}")
tok, ids, Xs = tiny_rows()
pieces = [tok.decode([i]) for i in ids]; k = pieces.index(" sat")
ln = LayerNorm(128)
z = ln(Xs)[k]
print(f"  Tiny's ' sat' entering the layer: length {np.linalg.norm(Xs[k]):.4f} -> after ln_1 length {np.linalg.norm(z):.3f}, first four {r3(z[:4], 3)}")

# Part 2: breakages
def stack(n=12, dtype=np.float32):
    return [(MultiHead(n_heads=4, seed=SEED + 100 * l), FeedForward(seed=SEED + 100 * l)) for l in range(n)]
layers = stack()
def cast(m, dt):
    for a in ("W_q", "W_k", "W_v", "W_o", "W_1", "W_2", "b_1", "b_2"):
        if hasattr(m, a): setattr(m, a, getattr(m, a).astype(dt))
for att, ff in layers: cast(att, np.float32); cast(ff, np.float32)
X32 = Xs.astype(np.float32)
def norm32(x):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return ((x - m) / np.sqrt(v + np.float32(1e-5))).astype(np.float32)

def run(x, residual=True, prenorm=True):
    lengths, sims = [], []
    for att, ff in layers:
        for f in (att, ff):
            h = f(norm32(x) if prenorm else x).astype(np.float32)
            x = (x + h) if residual else h
        lengths.append(float(np.linalg.norm(x.astype(np.float64), axis=1).mean())); sims.append(token_similarity(x))
    return x, lengths, sims

print("\nBreakage 1, the residual additions deleted (twelve untrained layers, float32):")
_, Lr, Sr = run(X32)
_, Lp, Sp = run(X32, residual=False)
_, Ln, Sn = run(X32, residual=False, prenorm=False)
print("  layer:                  " + " ".join(f"{i:>9}" for i in (1, 2, 3, 6, 9, 12)))
for name, Ls, Ss in (("correct (pre-norm)", Lr, Sr), ("no residual, pre-norm", Lp, Sp), ("no residual, no norm", Ln, Sn)):
    print(f"  {name:<22} length " + " ".join(f"{Ls[i - 1]:>9.3g}" for i in (1, 2, 3, 6, 9, 12)))
    print(f"  {'':<22} similar" + " ".join(f"{Ss[i - 1]:>9.3f}" if np.isfinite(Ss[i - 1]) else f"{'nan':>9}" for i in (1, 2, 3, 6, 9, 12)))
xn = run(X32, residual=False, prenorm=False)[0]
l32 = float(np.linalg.norm(xn, axis=1).mean())
print(f"  without residual or norm, the largest number left at layer 12 is {np.abs(xn).max():.1e}; the same rows' length "
      f"computed in float32 is {l32} (the squares fall below float32's smallest number): the signal has vanished")
x_last = np.array(Xs, dtype=np.float32); x_alt = x_last.copy()
x_alt[-1] = np.asarray(tiny_rows()[2][1], dtype=np.float32)     # replace the last token's row by ' cat''s
d_ok = np.abs(run(x_last)[0][-1] - run(x_alt)[0][-1]).max()
d_br = np.abs(run(x_last, residual=False)[0][-1] - run(x_alt, residual=False)[0][-1]).max()
print(f"  change the last token to ' cat': its top row moves by {d_ok:.3f} (correct) and {d_br:.1e} (no residual, pre-norm)")

def bad_norm_axis(x):
    m = x.mean(0, keepdims=True); v = ((x - m) ** 2).mean(0, keepdims=True)   # over the tokens
    return (x - m) / np.sqrt(v + 1e-5)
alt = Xs.copy(); alt[-1] = Xs[1]
leak = np.abs(bad_norm_axis(Xs)[:-1] - bad_norm_axis(alt)[:-1]).max()
ok = np.abs(layer_norm(Xs)[:-1] - layer_norm(alt)[:-1]).max()
print("\nBreakage 2, the average and spread taken over the tokens instead of over each row:")
print(f"  last token changed ('at' -> ' cat'): earlier rows move by {leak:.3f} (correct norm: {ok:.1e})")
b = bad_norm_axis(Xs)
print(f"  the output looks normal: shape {b.shape}, mean row length {np.linalg.norm(b, axis=1).mean():.2f} against {np.linalg.norm(layer_norm(Xs), axis=1).mean():.2f}")

def bad_ddof(x):
    m = x.mean(-1, keepdims=True); return (x - m) / np.sqrt(np.var(x, -1, ddof=1, keepdims=True) + 1e-5)
lst = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
print("\nBreakage 3, the sample spread (divide by d - 1):")
print(f"  the Interlude's list -> {r3(bad_ddof(lst), 3)} instead of {r3(layer_norm(lst), 3)}")
print(f"  Tiny's rows: every number {np.abs(bad_ddof(Xs) / layer_norm(Xs)).mean():.4f} times too small on average (sqrt(127/128) = {np.sqrt(127 / 128):.4f})")

print("\nPart 3. Checkpoint tests")
tests = [
    ("sat's toy row normalises to (-1, 1) and RMSNorm to (0.683, 1.238)", np.allclose(layer_norm(sat), [-1, 1], atol=1e-4) and np.allclose(rms_norm(sat), [0.683, 1.238], atol=1e-3)),
    ("Tiny's ' sat' after ln_1: length 11.191, first number -0.86", abs(np.linalg.norm(z) - 11.191) < 1e-3 and abs(z[0] + 0.86) < 5e-3),
    ("the Interlude's list normalises to (-1.5, -0.5, -0.5, -0.5, 0, 0, 1, 2) (catches Breakage 3)", np.allclose(layer_norm(lst), [-1.5, -0.5, -0.5, -0.5, 0, 0, 1, 2], atol=1e-4)),
    ("Breakage 3 fails that test", not np.allclose(bad_ddof(lst), [-1.5, -0.5, -0.5, -0.5, 0, 0, 1, 2], atol=1e-4)),
    ("changing the last token leaves earlier rows' norm output unchanged (catches Breakage 2)", ok == 0.0 and leak > 0),
    ("after twelve layers the last token still matters: its top row moves by more than 0.1 (catches Breakage 1)", d_ok > 0.1 and d_br < 0.01 and min(Lr) > 0),
]
for i, (name, passed) in enumerate(tests, 1):
    print(f"  {i}. {name}: {'pass' if passed else 'FAIL'}")
print("all checkpoint tests pass:", all(p for _, p in tests))
