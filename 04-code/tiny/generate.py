"""tiny/generate.py (Chapter 27): turning Tiny's next-token scores into text.

  next_scores(tt, ids)            raw next-token scores (float64 NumPy) for the last position, context cut to 1,024
  penalise(s, ids, rep, presence, frequency, no_repeat)
                                  repetition controls on the scores: rep > 1 divides positive scores and multiplies
                                  negative ones for tokens already written (the CTRL rule); presence subtracts a fixed
                                  amount from any token already written, frequency subtracts it once per occurrence;
                                  no_repeat = n forbids any token that would complete an n-token run already written
  shares(s, T, top_k, top_p, min_p, allowed)
                                  temperature, then the three tail cuts, then an optional set of allowed tokens;
                                  returns (shares, how many tokens were kept)
  generate(tt, ids, n, rng, ...)  the loop: score, adjust, choose (argmax if rng is None), append; stops after n
                                  tokens or when a token in `stop` is chosen.  Returns (ids, info)
  beam_search(tt, ids, n, width)  keeps the `width` partial continuations with the largest total log-probability
  option_logprobs(tt, ids, options)
                                  total log-probability of each whole option (a list of token lists) after ids
  local_mask_probs(tt, ids, options)
                                  the chance that token-by-token masking to the options' prefixes ends in each
                                  option (what a constrained sampler actually draws from)
  distinct_n(seqs, n), repeated_runs(ids, n)
                                  diversity across samples, repetition within one sample
Every draw comes from the generator passed in, normally tiny/seed.py's streams(seed)["sample"]."""
import numpy as np


def next_scores(tt, ids, ctx=1024):
    import torch
    with torch.no_grad():
        return tt.scores(list(ids)[-ctx:])[-1].double().numpy()


def penalise(s, ids, rep=1.0, presence=0.0, frequency=0.0, no_repeat=None):
    s = s.copy()
    ids = list(ids)
    if rep != 1.0 or presence or frequency:
        seen, counts = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True)
        if rep != 1.0:
            v = s[seen]
            s[seen] = np.where(v > 0, v / rep, v * rep)
        s[seen] -= presence + frequency * counts
    if no_repeat and len(ids) >= no_repeat:
        n = no_repeat
        head = tuple(ids[-(n - 1):]) if n > 1 else ()
        banned = {ids[i + n - 1] for i in range(len(ids) - n + 1) if tuple(ids[i:i + n - 1]) == head}
        s[list(banned)] = -np.inf
    return s


def shares(s, T=1.0, top_k=None, top_p=None, min_p=None, allowed=None):
    z = s / T
    z = z - z[np.isfinite(z)].max()
    q = np.exp(z)
    q /= q.sum()
    keep = q > 0
    if allowed is not None:
        m = np.zeros_like(keep); m[list(allowed)] = True; keep &= m
        q = np.where(keep, q, 0.0); q /= q.sum()
    if top_k is not None:
        keep &= q >= np.sort(q)[-top_k]
    if top_p is not None:
        o = np.argsort(-q)
        m = np.zeros_like(keep); m[o[:int(np.searchsorted(np.cumsum(q[o]), top_p) + 1)]] = True
        keep &= m
    if min_p is not None:
        keep &= q >= min_p * q.max()
    q = np.where(keep, q, 0.0)
    return q / q.sum(), int(keep.sum())


def generate(tt, ids, n, rng=None, T=1.0, top_k=None, top_p=None, min_p=None, rep=1.0, presence=0.0,
             frequency=0.0, no_repeat=None, stop=(), allowed=None):
    ids = list(ids)
    start = len(ids)
    info = dict(kept=[], surprise=[], stopped=False)
    for _ in range(n):
        raw = next_scores(tt, ids)
        s = penalise(raw, ids, rep, presence, frequency, no_repeat) if (rep != 1.0 or presence or frequency
                                                                         or no_repeat) else raw
        if rng is None:
            q, k = shares(s, 1.0, allowed=allowed)
            nxt = int(np.argmax(q)); k = 1
        else:
            q, k = shares(s, T, top_k, top_p, min_p, allowed)
            nxt = int(rng.choice(len(q), p=q))
        p0 = np.exp(raw - raw.max()); p0 /= p0.sum()
        info["kept"].append(k); info["surprise"].append(float(-np.log(p0[nxt])))
        ids.append(nxt)
        if nxt in stop:
            info["stopped"] = True
            break
    return ids, info


def beam_search(tt, ids, n, width):
    beams = [(0.0, list(ids))]
    for _ in range(n):
        cand = []
        for lp, b in beams:
            s = next_scores(tt, b)
            logp = s - s.max(); logp = logp - np.log(np.exp(logp).sum())
            for j in np.argsort(-logp)[:width]:
                cand.append((lp + float(logp[j]), b + [int(j)]))
        cand.sort(key=lambda x: -x[0])
        beams = cand[:width]
    return beams


def _logp(tt, ids):
    s = next_scores(tt, ids)
    z = s - s.max()
    return z - np.log(np.exp(z).sum())


def option_logprobs(tt, ids, options):
    out = []
    for o in options:
        lp, ctx = 0.0, list(ids)
        for tkn in o:
            lp += float(_logp(tt, ctx)[tkn]); ctx.append(tkn)
        out.append(lp)
    return np.array(out)


def local_mask_probs(tt, ids, options):
    """Exact chance that masking each step to tokens that continue some option, then drawing from the
    renormalised shares, ends in each option.  Options must not be prefixes of one another."""
    def walk(prefix, live, chance, res):
        done = [i for i in live if len(options[i]) == len(prefix)]
        for i in done:
            res[i] += chance
        live = [i for i in live if len(options[i]) > len(prefix)]
        if not live:
            return
        lp = _logp(tt, list(ids) + prefix)
        nexts = sorted({options[i][len(prefix)] for i in live})
        w = np.exp(lp[nexts]); w /= w.sum()
        for tkn, wt in zip(nexts, w):
            walk(prefix + [tkn], [i for i in live if options[i][len(prefix)] == tkn], chance * wt, res)
    res = np.zeros(len(options))
    walk([], list(range(len(options))), 1.0, res)
    return res


def distinct_n(seqs, n):
    grams = [tuple(s[i:i + n]) for s in seqs for i in range(len(s) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def repeated_runs(ids, n=4):
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    seen, r = set(), 0
    for g in grams:
        r += g in seen; seen.add(g)
    return r / max(1, len(grams))


# ---- Chapter 28 additions: rules that read the entropy, contrastive decoding, exact constrained choice ----
def probs(s):
    z = np.exp(s - s[np.isfinite(s)].max()); return z / z.sum()


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def typical_shares(s, tau):
    """Locally typical sampling: keep tokens in order of |surprise - entropy| until their shares reach tau."""
    p = probs(s)
    lp = np.log(np.clip(p, 1e-300, None)); H = -(p * lp).sum()
    o = np.argsort(np.abs(-lp - H))
    n = min(len(p), int(np.searchsorted(np.cumsum(p[o]), tau) + 1))
    q = np.zeros_like(p); q[o[:n]] = p[o[:n]]
    return q / q.sum(), n


def eta_shares(s, eps):
    """eta-sampling: keep tokens whose share exceeds min(eps, sqrt(eps) * exp(-entropy))."""
    p = probs(s)
    eta = min(eps, np.sqrt(eps) * np.exp(-entropy(p)))
    keep = p > eta
    if not keep.any():
        keep = p == p.max()
    q = np.where(keep, p, 0.0)
    return q / q.sum(), int(keep.sum())


def mirostat_step(s, mu):
    """Mirostat v2 cut: keep tokens with surprise below mu (the top token if none).  Caller updates
    mu -= lr * (surprise of the chosen token - target)."""
    p = probs(s)
    keep = -np.log(np.clip(p, 1e-300, None)) < mu
    if not keep.any():
        keep = p == p.max()
    q = np.where(keep, p, 0.0)
    return q / q.sum(), int(keep.sum())


def contrastive_scores(s_exp, s_ama, alpha=0.1, beta=1.0):
    """Contrastive decoding: among tokens with expert share >= alpha * the expert's top share (the plausibility
    cut), score log p_expert - beta * log p_amateur; every other token gets -inf.  Returns (scores, kept)."""
    le = np.log(np.clip(probs(s_exp), 1e-300, None)); la = np.log(np.clip(probs(s_ama), 1e-300, None))
    keep = le >= np.log(alpha) + le.max()
    out = np.where(keep, le - beta * la, -np.inf)
    return out, int(keep.sum())


def exact_option_shares(tt, ids, options):
    """The model's own distribution over whole options, conditioned on the output being one of them
    (what constrained decoding should sample from)."""
    lp = option_logprobs(tt, ids, options)
    w = np.exp(lp - lp.max())
    return w / w.sum()
