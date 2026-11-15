"""Chapter 11 exercises: worked numbers for Appendix A.  Run from 04-code/ch11:  python3 solutions.py"""
from _common import *
import time
from tiny.attention import softmax, causal_mask, tiled_attention

print("Exercise 1. Online softmax on the row [1, 3, 2]")
s = np.array([1.0, 3.0, 2.0]); m, l = -np.inf, 0.0
for x in s:
    mn = max(m, x); l = l * np.exp(m - mn) + np.exp(x - mn); m = mn
    print(f"  score {x:.0f}: running max {m:.0f}, running sum {l:.4f}")
print("  shares", r3(np.exp(s - m) / l), "| softmax", r3(softmax(s)))

print("\nExercise 2. Blocks of two, scores [0, 2, 1, 4], values (1,0), (0,1), (1,1), (2,0)")
s = np.array([0.0, 2.0, 1.0, 4.0]); Vv = np.array([[1, 0], [0, 1], [1, 1], [2, 0]], dtype=float)
m, l, o = -np.inf, 0.0, np.zeros(2)
for b in range(2):
    blk = s[2 * b:2 * b + 2]; mn = max(m, blk.max()); sc = np.exp(m - mn)
    p = np.exp(blk - mn); l = l * sc + p.sum(); o = o * sc + p @ Vv[2 * b:2 * b + 2]; m = mn
    print(f"  block {b}: max {m:.0f} | rescale {sc:.4f} | sum {l:.4f} | unnormalised output {r3(o)}")
print("  output", r3(o / l), "| reference", r3(softmax(s) @ Vv))

print("\nExercise 3. One head of width 128, 8,192 tokens, 16-bit numbers, query blocks of 128")
d, B, n, Br = 128, 2, 8192, 128
flops = 4 * n * n * d; plain = 4 * n * d * B + 4 * n * n * B; tiled = 2 * n * d * B + (n // Br) * 2 * n * d * B
print(f"  FLOPs {flops:,} | plain bytes {plain:,} ({plain / 2**20:,.0f} MiB) | tiled bytes {tiled:,} ({tiled / 2**20:,.0f} MiB)")
print(f"  operations per byte: plain {flops / plain:.1f} | tiled {flops / tiled:.1f}")

print("\nExercise 4. Blocks computed")
T = 2048 // 128
print(f"  causal, 2,048 tokens, blocks of 128: {T * (T + 1) // 2} of {T * T}")
TT = 8192 // 128; w = 512
win = sum(min(i + 1, w // 128 + 1) for i in range(TT))
print(f"  window 512, 8,192 tokens, blocks of 128: {win} of {TT * TT} (causal alone: {TT * (TT + 1) // 2})")

print("\nExercise 5. A machine of 1,000 trillion operations a second and 8 TB/s of main memory")
bal = 1000e12 / 8e12
print(f"  balance {bal:.0f} operations per byte; at width 128 plain attention does {flops / plain:.0f} per byte (limited by memory), tiled {flops / tiled:.0f} (just above the balance)")

print("\nExercise 6. Fixed shift for scores up to 20 with 16-bit exponentials")
lim = float(np.log(np.finfo(np.float16).max))
print(f"  float16 exp overflows above {lim:.2f}; the shift must be at least {20 - lim:.2f}")

print("\nExercise 7. tiled_attention block sizes at 2,048 tokens, width 64 (fastest of 5)")
rng = np.random.default_rng(SEED)
Q, K, V = (rng.normal(size=(2048, 64)) for _ in range(3))
def measure():
    res = {}
    for b in (32, 64, 128, 256, 512):
        best = 1e9
        for _ in range(5):
            t = time.perf_counter(); tiled_attention(Q, K, V, block=b); best = min(best, time.perf_counter() - t)
        res[str(b)] = round(best * 1000, 1)
    return res
r = recorded("solutions_timing", measure)
print(f"  measured {r['measured']}: " + " | ".join(f"block {b}: {r[str(b)]} ms" for b in (32, 64, 128, 256, 512)))

print("\nExercise 8. Keep m + log(l) per row and rebuild the share table, 64 random tokens")
q, k = rng.normal(size=(64, 16)), rng.normal(size=(64, 16))
S = np.where(causal_mask(64), q @ k.T / 4.0, -np.inf)
m = np.full(64, -np.inf); l = np.zeros(64)
for j0 in range(0, 64, 16):
    s = S[:, j0:j0 + 16]; mn = np.maximum(m, s.max(1)); mu = np.where(np.isinf(mn), 0, mn)
    l = l * np.exp(m - mu) + np.exp(s - mu[:, None]).sum(1); m = mn
lse = m + np.log(l)
P = np.exp(S - lse[:, None])
print(f"  rebuilt equals softmax: {bool(np.allclose(P, softmax(S)))} | numbers kept per row: 1")

print("\nExercise 9. Conditional rescaling with threshold log 16 (random scores, as rescale.py)")
Qr = rng.normal(size=(256, 64)); Kr = rng.normal(size=(4096, 64))
S = Qr @ Kr.T / 8.0
for tau, name in [(0.0, "always"), (np.log(16), "log 16"), (np.log(256), "log 256")]:
    m = np.full(256, -np.inf); resc = 0; big = 0.0
    for j0 in range(0, 4096, 64):
        mb = S[:, j0:j0 + 64].max(1); grow = mb > m + tau
        if j0 > 0: resc += int(grow.sum())
        m = np.where(grow, np.maximum(m, mb), m)
        big = max(big, float(np.exp(S[:, j0:j0 + 64] - m[:, None]).max()))
    print(f"  threshold {name:>7}: rescales {resc:,} of {256 * 63:,} | largest exponential held {big:.2f}")

print("\nExercise 10. Blocks of one (pure online softmax) on Chapter 8's hand head")
from tiny.attention import attention
out, _ = attention(X, W_q, W_k, W_v)
print("  tiled_attention(block=1) equals attention():", bool(np.allclose(tiled_attention(X @ W_q, X @ W_k, X @ W_v, block=1), out)))
