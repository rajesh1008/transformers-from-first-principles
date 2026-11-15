"""Learning embeddings from the novel with one rule: pull together words that appear near
each other, push apart random pairs.  (A small version of the classic skip-gram method;
Chapter 15 explains why the rule works.)  NumPy only; seeded, so every run is identical."""
import _common
import numpy as np
from collections import Counter
from tiny.corpus import load_text, words

def squash(x):
    """Turn any agreement into a number between 0 and 1: 'how sure are we these are neighbours?'"""
    return 1 / (1 + np.exp(-x))

def nudge(a, b, together, step=0.05):
    """One nudge of row a toward row b (together=True) or away from it (together=False).
    The surer we already are, the smaller the nudge."""
    sure = squash(dot := float(np.dot(a, b)))
    amount = (1 - sure) if together else -sure
    return a + step * amount * b, dot, sure

def corpus(min_count=5):
    w = words(load_text())
    counts = Counter(w)
    vocab = [x for x, n in counts.most_common() if n >= min_count]
    index = {x: i for i, x in enumerate(vocab)}
    ids = np.array([index.get(x, -1) for x in w])
    return w, counts, vocab, index, ids

def neighbour_pairs(ids, window=2):
    """Every (word, neighbour) pair within `window` places, both ways round; rare words skipped."""
    pairs = []
    for k in range(1, window + 1):
        a, b = ids[:-k], ids[k:]
        keep = (a >= 0) & (b >= 0)
        pairs += [np.stack([a[keep], b[keep]], 1), np.stack([b[keep], a[keep]], 1)]
    return np.concatenate(pairs)

def train(d=16, epochs=10, seed=1865, window=2, negatives=5, step=0.05, batch=256, track=None):
    """Two tables: A holds each word's row; B holds its row *as a neighbour*.  Returns A
    (and, if `track` lists word pairs, their cosine in A after every epoch)."""
    w, counts, vocab, index, ids = corpus()
    rng = np.random.default_rng(seed)
    V = len(vocab)
    A = rng.normal(0, 0.1, (V, d))                  # start: small random numbers
    B = np.zeros((V, d))
    p = np.array([counts[x] for x in vocab], float) ** 0.75
    p /= p.sum()                                    # random partners, common words more often
    P = neighbour_pairs(ids, window)
    cos = lambda x, y: float(A[x] @ A[y] / np.linalg.norm(A[x]) / np.linalg.norm(A[y]))
    history = [[cos(index[x], index[y]) for x, y in track]] if track else None
    for _ in range(epochs):
        P = P[rng.permutation(len(P))]
        for s in range(0, len(P), batch):
            wi, ci = P[s:s + batch, 0], P[s:s + batch, 1]
            ni = rng.choice(V, size=(len(wi), negatives), p=p)
            a = A[wi]
            pull = (1 - squash((a * B[ci]).sum(1)))[:, None]                  # real neighbours
            push = -squash(np.einsum("bd,bkd->bk", a, B[ni]))[:, :, None]      # random partners
            da = pull * B[ci] + (push * B[ni]).sum(1)
            np.add.at(B, ci, step * pull * a)
            np.add.at(B, ni, step * push * a[:, None, :])
            np.add.at(A, wi, step * da)
        if track: history.append([cos(index[x], index[y]) for x, y in track])
    return (A, vocab, index, history) if track else (A, vocab, index)

def nearest(A, vocab, index, word, k=6, exclude=()):
    N = A / np.linalg.norm(A, axis=1, keepdims=True)
    v = A[index[word]] if isinstance(word, str) else word
    s = N @ (v / np.linalg.norm(v))
    out = []
    for i in np.argsort(-s):
        if vocab[i] in exclude or (isinstance(word, str) and vocab[i] == word): continue
        out.append((vocab[i], round(float(s[i]), 2)))
        if len(out) == k: break
    return out
