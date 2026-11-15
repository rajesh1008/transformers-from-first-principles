"""Chapter 10 exercise solutions (Appendix A).  Run from 04-code/ch10:  python3 solutions.py"""
from _common import *
from tiny.attention import softmax, causal_mask, window_mask

print("Exercise 1: 32 layers, 32 query heads, 8 kv heads of 128, 16-bit numbers")
per_tok = 32 * 8 * 128 * 2 * 2
print(f"  grouped-query: {per_tok:,} bytes per token = {per_tok / 1024:.0f} KiB; at 32,768 tokens {per_tok * 32768 / 2**30:.1f} GiB")
print(f"  multi-head (32 kv heads): {per_tok * 4 / 1024:.0f} KiB per token; at 32,768 tokens {per_tok * 4 * 32768 / 2**30:.1f} GiB")

print("\nExercise 2: window 4, six layers: furthest seat back the last token can draw on =", 6 * (4 - 1))

print("\nExercise 3: window 2 on the running sentence, mat's row")
S = (X @ W_q) @ (X @ W_k).T / np.sqrt(2)
print("  mat's scores against 'the' (seat 4) and itself:", r3(S[5, 4:]))
A = softmax(np.where(window_mask(6, 2), S, -np.inf))
print("  shares:", r3(A[5, 4:]), " output:", r3((A @ (X @ W_v))[5]))

print("\nExercise 4: allowed pairs, n=6, window 2, global seat 0")
M = window_mask(6, 2).copy(); C = causal_mask(6)
M[:, 0] |= C[:, 0]
print(M.astype(int)); print("  allowed:", int(M.sum()), "of", int(C.sum()))

print("\nExercise 5: 8 query heads, 2 kv heads: kv block used by each query head")
print("  repeat (correct):", np.repeat(np.arange(2), 4).tolist(), " tile (wrong):", np.tile(np.arange(2), 4).tolist())

print("\nExercise 6: linear attention with phi(x) = relu(x) + 0.01, parallel vs recurrent")
phi = lambda x: np.maximum(x, 0) + 0.01
Q, K, V = phi(X @ W_q), phi(X @ W_k), X @ W_v
Wt = np.where(C, Q @ K.T, 0.0); par = (Wt / Wt.sum(1, keepdims=True)) @ V
Sx = np.zeros((2, 2)); z = np.zeros(2); rec = []
for t in range(6):
    Sx += np.outer(K[t], V[t]); z += K[t]; rec.append(Q[t] @ Sx / (Q[t] @ z))
print("  agree:", bool(np.allclose(par, rec)), " mat's output:", r3(par[5]))

print("\nExercise 7: tree toy, read about 12.5% of 4,096 keys with different group sizes")
rng = np.random.default_rng(SEED)
n, d = 4096, 32
Kk = rng.normal(size=(n, d)); Vv = rng.normal(size=(n, d))
needles = rng.choice(n - 64, size=8, replace=False)
q = rng.normal(size=d) * 0.3
for j in needles:
    Kk[j] = q / np.linalg.norm(q) * 6 + rng.normal(size=d) * 0.1
full = softmax(Kk @ q / np.sqrt(d)) @ Vv
for Cg in (8, 16, 32, 64):
    P = n // Cg
    k = max(1, (512 - P) // Cg)
    Kp = Kk.reshape(P, Cg, d).mean(1); Vp = Vv.reshape(P, Cg, d).mean(1)
    top = np.argsort(-(Kp @ q))[:k]; rest = np.setdiff1d(np.arange(P), top)
    idx = np.concatenate([np.arange(t * Cg, (t + 1) * Cg) for t in top])
    sc = np.concatenate([Kk[idx] @ q, Kp[rest] @ q]) / np.sqrt(d); sc[len(idx):] += np.log(Cg)
    out = softmax(sc) @ np.concatenate([Vv[idx], Vp[rest]])
    print(f"  C={Cg:>2}: parents {P:>3}, opened {k:>3}, entries read {P + len(idx):>4}, error {np.linalg.norm(out - full) / np.linalg.norm(full):.3f},"
          f" needles opened {len(set(needles) & set(idx.tolist()))}/8")

print("\nExercise 8: GPT-2 small's shape, 65,536 tokens, 1 full layer in 4, full layers with 2 kv heads, window layers 12")
full_b = 3 * 65536 * (2 * 64 * 2 * 2); win_b = 9 * 256 * (12 * 64 * 2 * 2)
print(f"  full layers {full_b / 2**20:.1f} MiB + window layers {win_b / 2**20:.2f} MiB = {(full_b + win_b) / 2**20:.1f} MiB")

print("\nExercise 9: window attention that never computes a masked score")
from tiny.attention import attention
def window_attention(X, W_q, W_k, W_v, w):
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    out = np.zeros((len(X), V.shape[1]))
    for i in range(len(X)):
        lo = max(0, i - w + 1)
        s = K[lo:i + 1] @ Q[i] / np.sqrt(K.shape[1])
        out[i] = softmax(s) @ V[lo:i + 1]
    return out
rng2 = np.random.default_rng(SEED)
Xr = rng2.normal(size=(300, 16)); Wq, Wk, Wv = (rng2.normal(size=(16, 8)) for _ in range(3))
Qr, Kr, Vr = Xr @ Wq, Xr @ Wk, Xr @ Wv
ref = softmax(np.where(window_mask(300, 32), Qr @ Kr.T / np.sqrt(8), -np.inf)) @ Vr
print(f"  largest difference from the masked version, 300 rows, window 32: {np.abs(window_attention(Xr, Wq, Wk, Wv, 32) - ref).max():.1e}")
print(f"  scores computed: {sum(min(i + 1, 32) for i in range(300)):,} instead of {300 * 300:,}")

print("\nExercise 10: recall of a plain linear state at widths 32 and 64 (first n with recall below 0.9)")
for d in (16, 32, 64):
    r = np.random.default_rng(SEED)
    first = None
    for n in (4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256):
        keys = r.normal(size=(n, d)); keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        vals = r.normal(size=(n, d))
        got = keys @ (keys.T @ vals)
        acc = float((((got[:, None] - vals[None]) ** 2).sum(-1).argmin(1) == np.arange(n)).mean())
        if acc < 0.9 and first is None:
            first = (n, acc)
    print(f"  width {d}: first below 0.9 at n = {first[0]} (recall {first[1]:.2f})")
