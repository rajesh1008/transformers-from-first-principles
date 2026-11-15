"""Chapter 24, section 24.2: what a loss number means.  Average surprise (natural log) of the true next token on the
Carroll held-out set's 16 windows (2,048 predictions) for four predictors that need no training of Tiny:
  uniform      every one of the 2,048 tokens gets the same share: ln 2048
  untrained    Tiny as Chapter 14 built it (seed 1865)
  unigram      each token's share of the Carroll pool (add-one smoothing)
  bigram       the share of each token after the previous one in the pool (add-k smoothing, k = 0.1, 0.01)
Also the surprise of the running sentence's seven next tokens under the uniform and unigram predictors."""
import _common
import math, json
import numpy as np, torch
torch.set_num_threads(1)
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from _pool import heldout_sets, corpora, POOL_BOOKS
from _common import sentence_ids, save_json
G, HB = heldout_sets()
c = corpora()
pool = np.concatenate([np.asarray(c["books"][b]) for b in POOL_BOOKS])
V = 2048
print(f"pool tokens {len(pool):,}; held-out windows {HB.shape}")
def surprise(P, W):
    """P(prev, next) -> share; average -ln share over every next token of the windows W."""
    tot, n = 0.0, 0
    for w in W:
        for a, b in zip(w[:-1], w[1:]):
            tot -= math.log(P(a, b)); n += 1
    return tot / n
uni = np.bincount(pool, minlength=V).astype(float)
pu = (uni + 1) / (uni.sum() + V)
big = np.zeros((V, V)); np.add.at(big, (pool[:-1], pool[1:]), 1)
out = {}
for name, W in (("carroll", HB), ("glass", G)):
    r = dict(uniform=math.log(V))
    tt = TorchTiny.from_numpy(Tiny(), dtype=torch.float32)
    with torch.no_grad():
        r["untrained"] = float(tt.loss(torch.tensor(W)))
    r["unigram"] = surprise(lambda a, b: pu[b], W)
    for k in (0.1, 0.01):
        rows = big.sum(1)
        r[f"bigram_k{k}"] = surprise(lambda a, b, k=k: (big[a, b] + k) / (rows[a] + k * V), W)
    out[name] = r
    print(name, "  ".join(f"{k} {v:.4f}" for k, v in r.items()))
s = sentence_ids()
print("running sentence", s, "unigram", round(surprise(lambda a, b: pu[b], [s]), 4),
      "bigram k0.01", round(surprise(lambda a, b: (big[a, b] + .01) / (big[a].sum() + .01 * V), [s]), 4))
print(f"tokens never seen in the pool: {(uni == 0).sum()} of {V}; pool unigram entropy {-(pu * np.log(pu)).sum():.4f}")
save_json("baselines", out)
