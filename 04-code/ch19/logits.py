"""Chapter 19, section 19.9: attention scores grow with the square of the query and key tables' size.
Layer 0 of Tiny, head 0, on eight windows of 129 tokens: the rows after ln_1 are fixed; W_q and W_k are
Tiny's own tables multiplied by k.  Reported: spread of the scores (after the 1/sqrt(32)), the average entropy of
the shares, the average largest share.  Then the same with QK-norm: each query and key row divided by its own
spread (RMS) before the dot product, which pins the score spread whatever k is."""
import _common
import numpy as np
from tiny.model import Tiny
from _common import novel_ids, tok, save_json, SEED, rms
T = tok(); nov = np.array(novel_ids(T))
starts = np.random.default_rng(SEED).integers(0, len(nov) - 129, size=8)
B = np.stack([nov[a:a + 128] for a in starts])
m = Tiny(); b = m.blocks[0]
X = np.stack([b.ln_1(m.embed(list(w))) for w in B])                         # (8, 128, 128)
Wq, Wk = b.att.W_q[:, :32], b.att.W_k[:, :32]
mask = np.tril(np.ones((128, 128), bool))
def stats(Q, K):
    S = Q @ np.swapaxes(K, -1, -2) / np.sqrt(32)
    Sm = np.where(mask, S, -np.inf)
    Sm = Sm - Sm.max(-1, keepdims=True); A = np.exp(Sm); A /= A.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(A > 0, A * np.log(A), 0).sum(-1)
    return float(rms(S[:, mask])), float(H[:, 1:].mean()), float(A[:, 1:].max(-1).mean())
def qk_norm(Z): return Z / np.sqrt((Z ** 2).mean(-1, keepdims=True) + 1e-6)
out = {}
print(f"uniform shares: average entropy {np.mean([np.log(t + 1) for t in range(1, 128)]):.3f}, average largest share "
      f"{np.mean([1 / (t + 1) for t in range(1, 128)]):.3f}")
print("  k     tables' spread   plain: score spread  entropy  largest share  |  QK-norm: score spread  entropy  largest share")
for k in (1, 3, 5, 10, 30):
    Q, K = X @ (Wq * k), X @ (Wk * k)
    p = stats(Q, K); q = stats(qk_norm(Q), qk_norm(K))
    out[str(k)] = dict(plain=p, qknorm=q)
    print(f"  {k:3d}   {0.02 * k:10.2f}       {p[0]:12.4g}   {p[1]:8.3f}  {p[2]:10.3f}     |  {q[0]:14.4g}   {q[1]:8.3f}  {q[2]:10.3f}")
save_json("logits", out)
