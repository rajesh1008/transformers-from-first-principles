"""One head of attention (Chapter 8).

Every token makes three new rows from its own row: a query (the question it asks), a key
(the label it wears) and a value (what it offers).  Scores are all the query-key dot products
at once, divided by the square root of the key width; softmax turns each row of scores into
shares that sum to one; the causal mask stops a token from looking at anything after it; the
output for each token is the share-weighted sum of the values.  NumPy only.
Chapter 9 adds the rotary option to Head and the class MultiHead, which splits the width into
several narrower heads and joins their outputs with an output table; Chapter 10 adds n_kv_heads
(grouped-query sharing of keys and values) and window (sliding-window attention) to MultiHead; Chapter 11 adds
tiled_attention (the same answer computed a block at a time, never holding the n x n table) and MultiHead(block=...);
Chapter 18 computes how to nudge the tables."""
import numpy as np


def softmax(scores, axis=-1):
    """Turn each row of scores into shares that sum to one.
    Subtracting the row's largest score first changes nothing in the answer (every exponential
    is divided by the same factor) and stops exp() from overflowing on large scores."""
    scores = np.asarray(scores, dtype=float)
    top = np.max(scores, axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)            # a row that is all -inf stays all zero
    e = np.exp(scores - top)
    return e / np.sum(e, axis=axis, keepdims=True)


def causal_mask(n):
    """An n x n table of True where a token may look (itself and earlier), False where it may not."""
    return np.tril(np.ones((n, n), dtype=bool))


def window_mask(n, window):
    """Chapter 10: True where a token may look under a sliding window: itself and the window - 1
    tokens before it, never the future."""
    i = np.arange(n)
    return (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - window)


def attention(X, W_q, W_k, W_v, causal=True, scale=True):
    """One head, start to finish.  X: (n, d) rows.  W_q, W_k: (d, d_k).  W_v: (d, d_v).
    Returns (out, shares): out is (n, d_v), one blended row per token; shares is (n, n)."""
    X = np.asarray(X, dtype=float)
    Q, K, V = X @ W_q, X @ W_k, X @ W_v                    # three lookups-by-multiplication
    scores = Q @ K.T                                       # all the dot products at once, (n, n)
    assert scores.shape == (len(X), len(X)), "score table must be tokens x tokens"   # Chapter 7's habit
    if scale:
        scores = scores / np.sqrt(K.shape[-1])             # keep the shares sane
    if causal:
        scores = np.where(causal_mask(len(X)), scores, -np.inf)   # no peeking
    shares = softmax(scores)                               # each row sums to one
    return shares @ V, shares                              # weighted sums of the values


def entropy(shares, axis=-1):
    """How spread out a row of shares is: 0 when all of it sits on one token, log(n) when even."""
    p = np.asarray(shares, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, -p * np.log(p), 0.0)
    return terms.sum(axis=axis)


def tiled_attention(Q, K, V, block=64, causal=True, window=None):
    """Chapter 11: attention computed a block at a time, with the same answer as softmax(Q K^T / sqrt(d)) V.
    Queries go in blocks of `block` rows, keys and values in blocks of `block` rows.  Each query block keeps
    three running quantities per row: the largest score so far (m), the sum of exponentials measured against
    it (l) and the unnormalised output (o).  When m grows, l and o are multiplied by exp(old m - new m).
    Key blocks wholly in the future (causal) or wholly outside the window are skipped before any score
    is computed.  The largest score table ever held is block x block.  Returns the (n, d_v) output only:
    the shares are never stored, which is the point."""
    Q, K, V = (np.asarray(T, dtype=float) for T in (Q, K, V))
    n, d = Q.shape
    out = np.empty((n, V.shape[1]))
    for i0 in range(0, n, block):
        q = Q[i0:i0 + block] / np.sqrt(d)
        rows = np.arange(i0, i0 + len(q))
        m = np.full(len(q), -np.inf); l = np.zeros(len(q)); o = np.zeros((len(q), V.shape[1]))
        for j0 in range(0, n, block):
            cols = np.arange(j0, min(j0 + block, n))
            if (causal or window is not None) and cols[0] > rows[-1]:
                break                                           # this block and all later ones are in the future
            if window is not None and cols[-1] <= rows[0] - window:
                continue                                        # wholly outside every row's window
            s = q @ K[j0:j0 + block].T
            ok = np.ones(s.shape, dtype=bool)
            if causal:
                ok &= cols[None, :] <= rows[:, None]
            if window is not None:                              # as window_mask: never the future either
                ok &= (cols[None, :] <= rows[:, None]) & (cols[None, :] > rows[:, None] - window)
            s = np.where(ok, s, -np.inf)
            m_new = np.maximum(m, s.max(1))
            m_use = np.where(np.isinf(m_new), 0.0, m_new)       # a row with nothing allowed yet stays empty
            scale = np.exp(m - m_use)                           # 1 when the max did not grow; 0 on the first block
            p = np.exp(s - m_use[:, None])
            l = l * scale + p.sum(1)
            o = o * scale[:, None] + p @ V[j0:j0 + block]
            m = m_new
        out[i0:i0 + len(q)] = o / l[:, None]
    return out


class Head:
    """One attention head for Tiny.  Three learned tables of d_model x d_head numbers each,
    drawn small and random (spread 0.02, as in GPT-2) and learned in training (Chapter 24).
    Seeds 1865 + 10, + 11, + 12, so that no table shares random numbers with the token table.
    Chapter 9: rotary=True turns each query and key by its position (Chapter 6's Rotary, sized
    to d_head) before the scores are taken; values are never turned."""

    def __init__(self, d_model=128, d_head=128, seed=1865, spread=0.02, causal=True, rotary=False):
        rng_q = np.random.default_rng(seed + 10)
        rng_k = np.random.default_rng(seed + 11)
        rng_v = np.random.default_rng(seed + 12)
        self.W_q = rng_q.normal(0.0, spread, size=(d_model, d_head))
        self.W_k = rng_k.normal(0.0, spread, size=(d_model, d_head))
        self.W_v = rng_v.normal(0.0, spread, size=(d_model, d_head))
        self.causal = causal
        self.rotary = None
        if rotary:
            from tiny.embed import Rotary
            self.rotary = Rotary(d=d_head)

    @property
    def n_params(self):
        return self.W_q.size + self.W_k.size + self.W_v.size

    def __call__(self, X, return_shares=False, start=0):
        if self.rotary is None:
            out, shares = attention(X, self.W_q, self.W_k, self.W_v, causal=self.causal)
        else:
            X = np.asarray(X, dtype=float)
            Q = self.rotary(X @ self.W_q, start=start)     # turn the questions by position
            K = self.rotary(X @ self.W_k, start=start)     # and the labels by the same clocks
            V = X @ self.W_v                               # what is offered does not turn
            scores = Q @ K.T / np.sqrt(K.shape[-1])
            if self.causal:
                scores = np.where(causal_mask(len(X)), scores, -np.inf)
            shares = softmax(scores)
            out = shares @ V
        return (out, shares) if return_shares else out


def split_heads(T, n_heads):
    """(n, n_heads * d_head) -> (n_heads, n, d_head): cut each row into n_heads equal pieces."""
    n, width = T.shape
    assert width % n_heads == 0, "the width must divide evenly into heads"
    return T.reshape(n, n_heads, width // n_heads).transpose(1, 0, 2)


def join_heads(T):
    """(n_heads, n, d_head) -> (n, n_heads * d_head): put the pieces of each row back side by side."""
    h, n, d_head = T.shape
    return T.transpose(1, 0, 2).reshape(n, h * d_head)


class MultiHead:
    """Several heads side by side (Chapter 9).  The query, key and value tables are each one
    d_model x (n_heads * d_head) table, cut into n_heads column blocks, one block per head; with
    n_heads=1 and d_head=128 they are exactly Chapter 8's Head tables (same seeds).  Each head
    scores, masks, softmaxes and blends inside its own d_head columns; the heads' outputs are put
    side by side and multiplied by the output table W_o, (n_heads * d_head) x d_model, seed + 13.
    `ablate` lists heads whose output is replaced before W_o: by zeros, or by `replace[h]`."""

    def __init__(self, d_model=128, n_heads=4, d_head=None, seed=1865, spread=0.02, causal=True,
                 rotary=False, n_kv_heads=None, window=None, block=None):
        self.block = block                                              # Chapter 11: tiled path when set
        d_head = d_model // n_heads if d_head is None else d_head
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads      # Chapter 10: grouped-query sharing
        assert n_heads % n_kv_heads == 0, "query heads must divide evenly into key-value groups"
        self.n_heads, self.d_head, self.d_model, self.n_kv_heads = n_heads, d_head, d_model, n_kv_heads
        width = n_heads * d_head
        self.W_q = np.random.default_rng(seed + 10).normal(0.0, spread, size=(d_model, width))
        # with n_kv_heads < n_heads the key and value tables keep only their first n_kv_heads blocks
        self.W_k = np.random.default_rng(seed + 11).normal(0.0, spread, size=(d_model, width))[:, :n_kv_heads * d_head]
        self.W_v = np.random.default_rng(seed + 12).normal(0.0, spread, size=(d_model, width))[:, :n_kv_heads * d_head]
        self.W_o = np.random.default_rng(seed + 13).normal(0.0, spread, size=(width, d_model))
        self.causal = causal
        self.window = window                                            # Chapter 10: look back at most window seats
        self.rotary = None
        if rotary:
            from tiny.embed import Rotary
            self.rotary = Rotary(d=d_head)

    @property
    def n_params(self):
        return self.W_q.size + self.W_k.size + self.W_v.size + self.W_o.size

    def head_outputs(self, X, start=0):
        """Every head at once: returns (outs, shares), outs (h, n, d_head), shares (h, n, n)."""
        X = np.asarray(X, dtype=float)
        n = len(X)
        Q = split_heads(X @ self.W_q, self.n_heads)
        K = split_heads(X @ self.W_k, self.n_kv_heads)
        V = split_heads(X @ self.W_v, self.n_kv_heads)
        if self.n_kv_heads < self.n_heads:                             # each key-value head serves a group of query heads
            group = self.n_heads // self.n_kv_heads
            K, V = np.repeat(K, group, axis=0), np.repeat(V, group, axis=0)
        if self.rotary is not None:
            Q = np.stack([self.rotary(q, start=start) for q in Q])
            K = np.stack([self.rotary(k, start=start) for k in K])
        if self.block is not None:                                     # Chapter 11: never hold the n x n tables
            outs = np.stack([tiled_attention(Q[h], K[h], V[h], block=self.block, causal=self.causal,
                                             window=self.window) for h in range(self.n_heads)])
            return outs, None
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_head)       # (h, n, n): one table per head
        assert scores.shape == (self.n_heads, n, n)
        if self.causal:
            scores = np.where(causal_mask(n), scores, -np.inf)
        if self.window is not None:
            scores = np.where(window_mask(n, self.window), scores, -np.inf)
        shares = softmax(scores)
        return shares @ V, shares

    def __call__(self, X, return_shares=False, start=0, ablate=(), replace=None):
        outs, shares = self.head_outputs(X, start=start)
        for h in ablate:                                   # remove a head *before* the output table
            outs[h] = 0.0 if replace is None else replace[h]
        out = join_heads(outs) @ self.W_o
        return (out, shares) if return_shares else out


# ---- Chapter 18: the backward pass of one head, by the rules of the chapter --------------------------------
def head_forward(A, W_q, W_k, W_v, W_o=None, causal=True):
    """One head, keeping every intermediate the backward pass needs.  A: (n, d) rows entering the head."""
    A = np.asarray(A, dtype=float)
    Q, K, V = A @ W_q, A @ W_k, A @ W_v
    c = 1.0 / np.sqrt(K.shape[-1])
    S = (Q @ K.T) * c
    if causal:
        S = np.where(causal_mask(len(A)), S, -np.inf)
    P = softmax(S)
    O = P @ V
    Y = O if W_o is None else O @ W_o
    return dict(A=A, W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o, Q=Q, K=K, V=V, c=c, S=S, P=P, O=O, Y=Y)


def softmax_backward(P, dP):
    """Through a row-wise softmax: dS = P * (dP - (sum over the row of P * dP)).  Every row of dS sums to zero."""
    return P * (dP - (P * dP).sum(-1, keepdims=True))


def head_backward(f, dY):
    """The nudge dY arriving at the head's output, back to every table.  f is head_forward's dict.
    Returns the nudges on O, P, S, Q, K, V, A and the four learned tables, each the shape of what it nudges."""
    g = {}
    if f["W_o"] is None:
        g["O"] = dY
    else:
        g["W_o"] = f["O"].T @ dY                       # Y = O W_o: X^T dY for the table ...
        g["O"] = dY @ f["W_o"].T                       # ... dY W^T for what came in
    g["P"] = g["O"] @ f["V"].T                         # O = P V
    g["V"] = f["P"].T @ g["O"]
    g["S"] = softmax_backward(f["P"], g["P"])          # through the softmax, row by row (masked shares are 0)
    g["Q"] = (g["S"] @ f["K"]) * f["c"]                # S = c Q K^T
    g["K"] = (g["S"].T @ f["Q"]) * f["c"]
    g["W_q"], g["W_k"], g["W_v"] = f["A"].T @ g["Q"], f["A"].T @ g["K"], f["A"].T @ g["V"]
    g["A"] = g["Q"] @ f["W_q"].T + g["K"] @ f["W_k"].T + g["V"] @ f["W_v"].T   # three uses of A: add
    return g


def tiled_head_backward(Q, K, V, O, lse, dO, block=64):
    """Chapter 18: the same nudges for Q, K and V without ever holding the n x n share table (as FlashAttention's
    backward does).  Kept from the forward pass: O and lse, each row's log of its sum of exponentials.  Each
    block of shares is recomputed as exp(S - lse); the row-sum term of softmax_backward is rowsum(dO * O),
    which equals rowsum(P * dP) because dP = dO V^T and O = P V.  Causal."""
    n, d = Q.shape
    c = 1.0 / np.sqrt(d)
    D = (dO * O).sum(-1)
    dQ, dK, dV = np.zeros_like(Q), np.zeros_like(K), np.zeros_like(V)
    for j0 in range(0, n, block):
        kj, vj = K[j0:j0 + block], V[j0:j0 + block]
        cols = np.arange(j0, j0 + len(kj))
        for i0 in range(j0, n, block):                     # causal: rows before j0 see none of these keys
            rows = np.arange(i0, min(i0 + block, n))
            s = (Q[rows] @ kj.T) * c
            s = np.where(cols[None, :] <= rows[:, None], s, -np.inf)
            p = np.exp(s - lse[rows, None])
            dV[j0:j0 + block] += p.T @ dO[rows]
            dp = dO[rows] @ vj.T
            ds = p * (dp - D[rows, None])
            dQ[rows] += (ds @ kj) * c
            dK[j0:j0 + block] += (ds.T @ Q[rows]) * c
    return dQ, dK, dV
