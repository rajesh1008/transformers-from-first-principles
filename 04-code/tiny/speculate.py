"""tiny/speculate.py (Chapter 32): speculative decoding for any model in tiny/load.py whose forward accepts
(ids, cache, start), with tiny/cache.py's cache.

  trim(cache, n)                 cut every layer's keys and values back to the first n positions (after a rejection)
  verify(model, cache, last, guess, start)
                                 one target pass over [last] + guess at positions start..; returns score rows [k + 1, V]
  accept_greedy(guess, rows)     number of leading guesses equal to the target's top choice, and the target's next token
  accept_sample(guess, q_rows, p_rows, rng)
                                 the speculative-sampling rule: accept g_i with probability min(1, p(g_i) / q(g_i));
                                 at the first rejection draw from max(0, p - q) renormalised; if every guess is
                                 accepted draw one more token from p.  q_rows[i] and p_rows[i] are probability vectors
                                 (numpy).  A deterministic draft has q one-hot on its guess.  Returns the new tokens.
  speculate(model, prompt, n, propose, k, sample=None)
                                 the whole loop.  propose(ids, k) -> (guesses, q_rows or None).  sample=None is greedy;
                                 otherwise sample=(temperature, rng).  Returns (ids, stats)
  lookup_draft(ids, k)           prompt lookup: copy what followed the last earlier match of the final two tokens
  Drafts that keep their own cache must trim it themselves (Chapter 32's EarlyExit does)."""
import numpy as np


def trim(cache, n):
    for l, c in enumerate(cache):
        if c is not None:
            cache[l] = (c[0][:, :n], c[1][:, :n])


def verify(model, cache, last, guess, start):
    import torch
    with torch.no_grad():
        return model.forward([last] + list(guess), cache, start)


def accept_greedy(guess, rows):
    tgt = rows.argmax(-1).tolist()
    m = 0
    while m < len(guess) and guess[m] == tgt[m]:
        m += 1
    return m, tgt[m]


def probs(row, T):
    z = row.double().numpy() / T
    z = np.exp(z - z.max())
    return z / z.sum()


def accept_sample(guess, q_rows, p_rows, rng):
    out = []
    for i, g in enumerate(guess):
        p, q = p_rows[i], q_rows[i]
        if rng.random() < min(1.0, p[g] / q[g]):
            out.append(g)
            continue
        r = np.maximum(p - q, 0.0)
        out.append(int(rng.choice(len(r), p=r / r.sum())))
        return out
    out.append(int(rng.choice(len(p_rows[-1]), p=p_rows[-1])))
    return out


def one_hot(V, g):
    q = np.zeros(V)
    q[g] = 1.0
    return q


def lookup_draft(ids, k, fallback):
    for i in range(len(ids) - 3, -1, -1):
        if ids[i] == ids[-2] and ids[i + 1] == ids[-1]:
            c = ids[i + 2:i + 2 + k]
            if c:
                return c + [c[-1]] * (k - len(c))
    return [fallback] * k


def speculate(model, prompt, n, propose, k, sample=None, on_accept=None):
    import torch
    from tiny.cache import new_cache
    ids = list(prompt); cache = new_cache(model)
    st = dict(passes=0, accepted=0, proposed=0)
    with torch.no_grad():
        s = model.forward(ids, cache, 0)[-1]; st["passes"] += 1
        if sample is None:
            ids.append(int(s.argmax()))
        else:
            T, rng = sample
            p = probs(s, T); ids.append(int(rng.choice(len(p), p=p)))
        while len(ids) - len(prompt) < n:
            guess, q_rows = propose(ids, k); st["proposed"] += k
            rows = verify(model, cache, ids[-1], guess, len(ids) - 1); st["passes"] += 1
            if sample is None:
                m, t = accept_greedy(guess, rows); new = guess[:m] + [t]
            else:
                p_rows = [probs(r, T) for r in rows]
                if q_rows is None:
                    q_rows = [one_hot(len(p_rows[0]), g) for g in guess]
                new = accept_sample(guess, q_rows, p_rows, rng); m = len(new) - 1
            st["accepted"] += m
            ids += new
            trim(cache, len(ids) - 1)
            if on_accept:
                on_accept(len(ids) - 1)
    return ids[:len(prompt) + n], st
