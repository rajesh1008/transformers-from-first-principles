"""Chapter 13, solutions to the numerical exercises (Appendix A).  Exercise 9 runs GPT-2 twelve times
(about two minutes).  Run from 04-code/ch13:  python3 solutions.py"""
from _common import *
from tiny.norm import layer_norm, rms_norm, LayerNorm, RMSNorm
import subprocess

print("1. (1, 3, 5, 7):", f"average {np.mean([1, 3, 5, 7]):.0f}, spread {np.std([1, 3, 5, 7]):.3f} (sqrt 5)",
      "->", r3(layer_norm([1, 3, 5, 7])))
print("2. RMSNorm of (3, 4): root mean square", f"{np.sqrt(12.5):.3f}", "->", r3(rms_norm([3, 4])))
A, Y = hand_layer(X)
print("3. cat", r3(Y[1]), "->", r3(layer_norm(Y[1])), "| mat", r3(Y[5]), "->", r3(layer_norm(Y[5])))
g = np.array([2.0, 0.5]); b = np.array([0.0, 1.0])
print("4. mat normalised then gain (2, 0.5), bias (0, 1):", r3(layer_norm(Y[5], g, b)))
a = 12
print(f"5. post-norm after six same-size unrelated additions: (1/sqrt 2)^6 = {2 ** -3:.4f}; "
      f"with the carried row weighted by a = {a}: (a / sqrt(a^2 + 1))^6 = {(a / np.sqrt(a * a + 1)) ** 6:.4f}")
print(f"6. GPT-2 medium (24 layers, width 1,024), layer norms: (2 x 24 + 1) x 2 x 1,024 = {(2 * 24 + 1) * 2 * 1024:,}; "
      f"RMSNorm model, 32 layers, width 4,096: (2 x 32 + 1) x 4,096 = {(2 * 32 + 1) * 4096:,}")

tok, ids, Xs = tiny_rows()
k = [tok.decode([i]) for i in ids].index(" sat")
z1, z2 = LayerNorm(128)(Xs)[k], RMSNorm(128)(Xs)[k]
print(f"7. Tiny's ' sat': layer norm against RMSNorm, largest difference {np.abs(z1 - z2).max():.4f}, "
      f"cosine {np.dot(z1, z2) / np.linalg.norm(z1) / np.linalg.norm(z2):.4f}")

# 8: post-norm with RMSNorm, spread 0.088, the carried copy
from tiny.attention import MultiHead
from tiny.ffn import FeedForward
_, _, X0 = tiny_rows(256)
sp = 1 / np.sqrt(128)
x = X0.copy(); c = np.ones((len(x), 1)); out = {}
for l in range(48):
    att, ff = MultiHead(n_heads=4, seed=SEED + 100 * l, spread=sp), FeedForward(seed=SEED + 100 * l, spread=sp)
    for f in (att, ff):
        z = x + f(x); q = np.sqrt((z ** 2).mean(-1, keepdims=True) + 1e-5); x = z / q; c = c / q
    if l + 1 in (1, 2, 4, 6, 12):
        out[l + 1] = float((c[:, 0] * np.linalg.norm(X0, axis=1) / np.linalg.norm(x, axis=1)).mean())
print("8. post-norm with RMSNorm, spread 0.088, carried share:", {k_: round(v, 3) for k_, v in out.items()})

# 10: eps
row = Xs[k]
for eps in (1e-12, 1e-5, 1e-2):
    zz = layer_norm(row, eps=eps)
    print(f"10. eps {eps:g}: ' sat' normalised spread {zz.std():.4f}, length {np.linalg.norm(zz):.3f}")

# 9: which ln_1, frozen alone, costs most (GPT-2)
import gpt2_norms_lib as G
base = G.forward()[0]["loss"]
fixed = G.fixed_divisors()
res = []
for l in range(12):
    nm = f"h.{l}.ln_1"
    r = G.forward({nm: ("f", fixed[nm], True)})[0]["loss"]
    res.append((l, r - base))
print("9. each layer's ln_1 frozen alone (seat 0 keeps its own), change in loss:")
print("   " + " | ".join(f"{l}: {d:+.3f}" for l, d in res))
print(f"   largest: layer {max(res, key=lambda t: t[1])[0]}")
