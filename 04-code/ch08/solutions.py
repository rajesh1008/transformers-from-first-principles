"""Chapter 8: numbers for the exercises (Appendix A).
Run from 04-code/ch08 after gpt2_head.py:  python3 solutions.py   (add --measure to retime exercise 12)"""
from _common import *
import time
from tiny.attention import attention, softmax, causal_mask, entropy, Head
from tiny.embed import TOY_POS, Embedding, PositionTable
from tiny.tokenizer import load_bpe

Q, K, V = X @ W_q, X @ W_k, X @ W_v
r = lambda a: np.round(np.asarray(a, dtype=float), 3).tolist()
print("1: cat query", r(Q[1]), "key", r(K[1]), "value", r(V[1]), "| cat->sat", Q[1] @ K[2], " sat->cat", Q[2] @ K[1])
print("2:", r(softmax([3, 1, 0])), r(softmax([10, 8, 7])), r(softmax([6, 2, 0])))
s_on = np.array([-1.414, 0.707, -0.707, -2.121]); print("3:", r(softmax(s_on)))
_, A = attention(X, W_q, W_k, W_v)
print("4:", r(A[3, :4] @ V[:4]))
print("5:", r(softmax([2, 2, 2, -np.inf])), softmax([2, 2, 2, -100]).tolist())
rng = np.random.default_rng(1865)
q, k = rng.normal(size=(2000, 64)), rng.normal(size=(2000, 64))
d = (q * k).sum(1)
print(f"6: spread of raw scores at width 64: {d.std():.2f}; divide by 8; after: {(d / 8).std():.2f}")
out, _ = attention(X, W_q, W_k, W_v)
seen = V                                           # mat sees every value
print("7: mat's output", r(out[5]), "first numbers seen", V[:, 0].min(), "to", V[:, 0].max(), "| second", V[:, 1].min(), "to", V[:, 1].max())
print("8: allowed entries", int(causal_mask(6).sum()), "of 36; for 1,024:", f"{int(1024 * 1025 / 2):,}", "of", f"{1024 * 1024:,}")

R = X + TOY_POS
Qp = R @ W_q
outp, Ap = attention(R, W_q, W_k, W_v)
print("9: the queries with positions", r(Qp[0]), r(Qp[4]), "| outputs", r(outp[0]), r(outp[4]), "| sat's shares", r(Ap[2, :3]))

def leak_test(head, X_, change_row):
    a = head(X_); X2 = X_.copy(); X2[-1] = change_row; b = head(X2)
    return [i for i in range(len(X_)) if not np.allclose(a[i], b[i])]
tok = load_bpe(TINY_BPE); emb, pos = Embedding(2048), PositionTable()
ids = tok.encode(SENTENCE); B = emb(ids) + pos(range(len(ids)))
other = emb(tok.encode(" hat"))[0] + pos([len(ids) - 1])[0]
print("10: rows that change, causal:", leak_test(Head(), B, other), " not causal:", leak_test(Head(causal=False), B, other))

for width in (2, 128):
    g = np.random.default_rng(1865)
    qq, kk = g.normal(size=(2000, width)), g.normal(size=(2000, 6, width))
    raw = np.einsum("nd,nkd->nk", qq, kk)
    for name, div in (("sqrt(d)", np.sqrt(width)), ("d", width)):
        sh = softmax(raw / div)
        print(f"11: width {width:>4}, divide by {name:>7}: top share {sh.max(1).mean():.3f}, entropy {entropy(sh).mean():.3f}")

def measure():
    h, g, res = Head(), np.random.default_rng(1865), {}
    for n in (256, 512, 1024, 2048):
        Xn = g.normal(0, 0.02, size=(n, 128)); h(Xn)
        t = time.perf_counter()
        for _ in range(5): h(Xn)
        res[str(n)] = (time.perf_counter() - t) / 5 * 1000
    return res
t = recorded("solutions_timing", measure)
print("12:", ", ".join(f"{n}: {t[n]:.2f} ms" for n in ("256", "512", "1024", "2048")), f"(measured {t['measured']})")

S = np.load(os.path.join(HERE, "outputs", "gpt2_shares.npy"))
prev = np.array([[np.mean([S[l, h, i, i - 1] for i in range(1, 6)]) for h in range(12)] for l in range(12)])
top = sorted(((prev[l, h], l, h) for l in range(12) for h in range(12)), reverse=True)[:6]
print("13: heads with average previous-token share above 0.5:", int((prev > 0.5).sum()), "of 144; strongest:",
      ", ".join(f"L{l}H{h} {v:.3f}" for v, l, h in top))
