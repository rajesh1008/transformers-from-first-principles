"""Chapter 10, section 6: linear attention and the running summary.  Replace softmax's exp(score) with a
product of positive feature maps, phi(q).phi(k); then the blend can be computed two ways that agree
exactly: as a masked table (parallel form) or as a running state S (d x d) and z (d) updated once per token
(recurrent form).  The state never grows.  Then the cost: a fixed-size state forgets when asked to store
many key-value pairs, while softmax attention over a cache does not.  Run:  python3 linear.py"""
from _common import *
from tiny.attention import softmax, causal_mask

def phi(x):                     # elu(x) + 1: positive, smooth, used by the 2020 linear-attention paper
    return np.where(x > 0, x + 1.0, np.exp(np.minimum(x, 0)))

Q, K, V = X @ W_q, X @ W_k, X @ W_v
# parallel form
Fq, Fk = phi(Q), phi(K)
Wt = np.where(causal_mask(6), Fq @ Fk.T, 0.0)
A_lin = Wt / Wt.sum(1, keepdims=True)
out_par = A_lin @ V
# recurrent form: one state, updated token by token
S = np.zeros((2, 2)); z = np.zeros(2); out_rec = []
for t in range(6):
    S = S + np.outer(Fk[t], V[t]); z = z + Fk[t]
    out_rec.append((Fq[t] @ S) / (Fq[t] @ z))
out_rec = np.array(out_rec)
A_soft = softmax(np.where(causal_mask(6), Q @ K.T / np.sqrt(2), -np.inf))
print("linear attention on the running sentence (Chapter 8's hand tables, phi = elu + 1):")
print(table(A_lin))
print("mat's row, linear:", r3(A_lin[5]), "| softmax (Chapter 8):", r3(A_soft[5]))
print("mat's output, linear:", r3(out_par[5]), "| softmax:", r3((A_soft @ V)[5]))
print("parallel and recurrent forms agree:", bool(np.allclose(out_par, out_rec)),
      "| largest difference", f"{np.abs(out_par - out_rec).max():.1e}")
print("state after mat: S =", r3(S), " z =", r3(z), "->", S.size + z.size, "numbers, whatever the length")

# sizes: what each design keeps per head per layer
print("\nwhat one head of width 64 keeps while writing (numbers):")
for n in (1024, 65536, 1_000_000):
    print(f"  after {n:>9,} tokens: key-value cache {2 * 64 * n:>12,} | linear state {64 * 64 + 64:>6,}")

# recall test: store n random key -> value pairs, then ask for each value back by its key
rng = np.random.default_rng(SEED)
d = 16
print(f"\nrecall test, width {d}, seed {SEED}: store n pairs, look each value up by its key;")
print("score = share of lookups whose answer is closest to the right value (chance = 1/n)")
for n in (4, 8, 16, 32, 64, 128):
    keys = rng.normal(size=(n, d)); keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    vals = rng.normal(size=(n, d))
    # softmax attention over the full cache, sharp (scores times 8)
    A = softmax(8 * keys @ keys.T); got_soft = A @ vals
    # plain linear state: S = sum k v^T  (no normaliser needed for this test)
    S = keys.T @ vals; got_lin = keys @ S
    def acc(got):
        dist = ((got[:, None, :] - vals[None, :, :]) ** 2).sum(-1)
        return float((dist.argmin(1) == np.arange(n)).mean())
    print(f"  n={n:>4}: softmax over cache {acc(got_soft):5.2f} | linear state {acc(got_lin):5.2f}"
          f" | cache numbers {2 * n * d:>5} vs state {d * d}")
