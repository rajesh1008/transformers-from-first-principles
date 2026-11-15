"""Chapter 25: evaluating a trained Tiny.  NumPy, with PyTorch only for the forward pass.

    load(path)                     a TorchTiny (32-bit) with the tables saved by Chapter 24's train_tiny.py
    windows(ids, n=129)            every non-overlapping window of n tokens (the last partial one dropped)
    score(tt, W, edges)            one forward pass per window; per prediction: surprise (-ln p of the true token),
                                   the true token, the top token and its probability, the entropy of the
                                   distribution; plus two calibration tallies over the probability bins `edges`:
                                   top-1 (confidence of the top token against whether it was right) and
                                   all-token (every one of the 2,048 probabilities, against whether that token came next)
    perplexity(loss)               e ** loss
    ece(tally)                     expected calibration error: the share-weighted average gap between average
                                   confidence and how often it came true, over the bins of a tally
    bootstrap(x, groups, B, seed)  the spread of a mean when whole groups (windows) are drawn again with replacement;
                                   returns (mean, low, high) for a 95 per cent interval and the standard error
    paired(a, b, groups, B, seed)  the same for the per-prediction difference a - b of two models on the same text
    sign_test(d)                   two-sided exact binomial test of how many groups favour one model (ties dropped)
    ngrams(ids, n), overlap(...)   token n-gram sets and the share of an evaluation text's n-grams found in training text

Every function takes plain arrays and returns floats, arrays or dicts, so each number can be checked by hand."""
import math
import numpy as np

EDGES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-9])

def load(path, dtype=None):
    import torch
    from tiny.model import Tiny
    from tiny.torch_tiny import TorchTiny
    tt = TorchTiny.from_numpy(Tiny(), dtype=dtype or torch.float32)
    sd = torch.load(path)
    for k in tt.t:
        tt.t[k] = sd[k].to(tt.dtype).detach()
    return tt

def windows(ids, n=129):
    ids = np.asarray(ids)
    k = (len(ids) - 1) // (n - 1)
    return np.stack([ids[i * (n - 1): i * (n - 1) + n] for i in range(k)])

def _tally(edges):
    nb = len(edges) - 1
    return dict(edges=list(map(float, edges)), n=np.zeros(nb), conf=np.zeros(nb), hit=np.zeros(nb))

def score(tt, W, edges=EDGES, temperature=1.0, batch=8):
    """W: (k, n) windows.  Seats 0 .. n-2 of each window predict 1 .. n-1."""
    import torch
    W = np.asarray(W)
    out = dict(surprise=[], target=[], top=[], ptop=[], entropy=[], window=[])
    top1, allt = _tally(edges), _tally(edges)
    with torch.no_grad():
        for s in range(0, len(W), batch):
            X = torch.as_tensor(W[s:s + batch], dtype=torch.long)
            S = tt.scores(X[:, :-1]).double() / temperature
            logp = torch.log_softmax(S, -1)
            P = logp.exp()
            tgt = X[:, 1:]
            lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            pt, top = P.max(-1)
            ent = -(P * logp).sum(-1)
            out["surprise"].append((-lp).numpy().ravel()); out["target"].append(tgt.numpy().ravel())
            out["top"].append(top.numpy().ravel()); out["ptop"].append(pt.numpy().ravel())
            out["entropy"].append(ent.numpy().ravel())
            out["window"].append(np.repeat(np.arange(s, s + len(X)), X.shape[1] - 1))
            b = np.clip(np.searchsorted(edges, pt.numpy().ravel(), side="right") - 1, 0, len(edges) - 2)
            right = (top == tgt).numpy().ravel()
            np.add.at(top1["n"], b, 1); np.add.at(top1["conf"], b, pt.numpy().ravel()); np.add.at(top1["hit"], b, right)
            Pf = P.reshape(-1, P.shape[-1]).numpy()
            bf = np.clip(np.searchsorted(edges, Pf, side="right") - 1, 0, len(edges) - 2)
            allt["n"] += np.bincount(bf.ravel(), minlength=len(edges) - 1)
            allt["conf"] += np.bincount(bf.ravel(), weights=Pf.ravel(), minlength=len(edges) - 1)
            hit_b = bf[np.arange(len(bf)), tgt.numpy().ravel()]
            allt["hit"] += np.bincount(hit_b, minlength=len(edges) - 1)
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["correct"] = out["top"] == out["target"]
    out["top1"], out["all"] = top1, allt
    return out

def perplexity(loss):
    return math.exp(loss)

def ece(t):
    n = np.asarray(t["n"], float); m = n > 0
    gap = np.abs(np.asarray(t["conf"])[m] / n[m] - np.asarray(t["hit"])[m] / n[m])
    return float((n[m] / n.sum() * gap).sum())

def reliability(t):
    n = np.asarray(t["n"], float)
    return [(t["edges"][i], t["edges"][i + 1], int(n[i]), float(t["conf"][i] / n[i]) if n[i] else None,
             float(t["hit"][i] / n[i]) if n[i] else None) for i in range(len(n))]

def _group_sums(x, groups):
    g = np.unique(groups, return_inverse=True)[1]
    return np.bincount(g, weights=x), np.bincount(g).astype(float)

def bootstrap(x, groups=None, B=2000, seed=1865):
    """Resample whole groups (windows) with replacement B times; the mean is over predictions."""
    x = np.asarray(x, float)
    if groups is None:
        groups = np.arange(len(x))
    s, c = _group_sums(x, groups)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(B, len(s)))
    means = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(mean=float(x.mean()), low=float(lo), high=float(hi), se=float(means.std(ddof=1)), groups=len(s))

def paired(a, b, groups=None, B=2000, seed=1865):
    return bootstrap(np.asarray(a, float) - np.asarray(b, float), groups, B, seed)

def sign_test(d):
    d = np.asarray(d, float); d = d[d != 0]
    n, k = len(d), int((d > 0).sum())
    lo = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(0, lo + 1)) / 2 ** n * 2
    return dict(n=n, positive=k, p=float(min(1.0, p)))

def ngrams(ids, n):
    ids = list(map(int, ids))
    return {tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)}

def overlap(train_set, ids, n):
    """Share of the n-grams in ids (positions, not distinct) that occur in train_set, and a per-position flag."""
    ids = list(map(int, ids))
    flags = np.array([tuple(ids[i:i + n]) in train_set for i in range(len(ids) - n + 1)])
    return float(flags.mean()) if len(flags) else 0.0, flags
