"""Tiny, assembled (Chapter 14).  Token table + position table, n_layers pre-norm Blocks, a final
layer norm ln_f, and the output: the next-token scores are the agreement of each top row with every
row of the token table (tied, Chapter 4), so the output adds no learned numbers of its own.

    rows   = E[ids] + P[0..n-1]                    (n, 128)
    rows   = Block_l(rows) for l = 0 .. 11         (n, 128)
    top    = ln_f(rows)                            (n, 128)
    scores = top @ E.T                             (n, 2048)   one score per token, per seat
    shares = softmax(scores)                       (n, 2048)   each row sums to one

Untrained: every table is random (seed 1865 and the per-layer rule in tiny/block.py), so the
shares are nonsense by design until Chapters 17 and 24 train it.  `forward(..., keep=True)` returns
the row entering every layer as well, for following a token up the stack.

Chapter 15: d_ff, activation and norm are passed to every Block (and norm to ln_f), so that
Tiny(vocab_size=12, n_layers=2, d_model=2, n_heads=1, d_ff=3, context=6, activation="relu", norm="rms")
is the hand-sized Tiny of the workshop chapter.  Defaults are Chapter 14's model, unchanged."""
import numpy as np
from tiny.embed import Embedding, PositionTable
from tiny.block import Block, SEED, NORMS
from tiny.norm import LayerNorm
from tiny.attention import softmax


class Tiny:
    def __init__(self, vocab_size=2048, n_layers=12, d_model=128, n_heads=4, context=1024, seed=SEED,
                 tied=True, final_norm=True, d_ff=None, activation="gelu", norm="layer"):
        self.E = Embedding(vocab_size, d=d_model, seed=seed)
        self.P = PositionTable(context=context, d=d_model, seed=seed)
        self.blocks = [Block(layer=l, d_model=d_model, n_heads=n_heads, seed=seed, d_ff=d_ff,
                             activation=activation, norm=norm) for l in range(n_layers)]
        self.ln_f = NORMS[norm](d_model) if final_norm else None
        self.tied = tied
        # an untied output table, if asked for, gets its own seed (Chapter 14 compares the counts)
        self.W_out = None if tied else np.random.default_rng(seed + 2).normal(0.0, 0.02, size=(d_model, vocab_size))
        self.vocab_size, self.d_model, self.n_layers, self.context = vocab_size, d_model, n_layers, context

    # ---- counting -------------------------------------------------------------------------------
    def count(self):
        """Learned numbers by part, counted from the tables themselves (unique tables only, so a
        table that appears twice is counted once)."""
        seen, parts = set(), {}
        def add(part, arr):
            if id(arr) in seen:
                return
            seen.add(id(arr))
            parts[part] = parts.get(part, 0) + arr.size
        add("token table", self.E.table)
        add("position table", self.P.table)
        for b in self.blocks:
            for name, arr in b.tables().items():
                add("layers: " + ("norms" if name.startswith("ln") else "attention" if name.startswith("att") else "feed-forward"), arr)
        if self.ln_f is not None:
            add("final norm", self.ln_f.gain)
            if hasattr(self.ln_f, "bias"):
                add("final norm", self.ln_f.bias)
        if self.W_out is not None:
            add("output table", self.W_out)
        return parts

    @property
    def n_params(self):
        return sum(self.count().values())

    # ---- running ------------------------------------------------------------------------------
    def embed(self, ids, start=0):
        ids = np.asarray(ids)
        return self.E(ids) + self.P(range(start, start + len(ids)))

    def top_rows(self, X):
        return self.ln_f(X) if self.ln_f is not None else X

    def scores_from(self, X):
        """Next-token scores for rows X (already through ln_f): (n, vocab)."""
        W = self.E.table.T if self.tied else self.W_out
        return X @ W

    def forward(self, ids, keep=False, n_layers=None, skip=()):
        """Scores for every seat.  keep=True also returns the rows entering each layer and the top
        rows (a list of n_layers + 1 tables).  skip = layers to leave out (a layer removed)."""
        X = self.embed(ids)
        stream = [X] if keep else None
        for l, b in enumerate(self.blocks[:n_layers]):
            if l not in skip:
                X = b(X)
            if keep:
                stream.append(X)
        S = self.scores_from(self.top_rows(X))
        return (S, stream) if keep else S

    def shares(self, ids):
        return softmax(self.forward(ids))

    def loss(self, ids):
        """Average surprise (natural log) of each true next token, seats 0 .. n-2 predicting 1 .. n-1.
        Chapter 24 names and explains this number; here it is only a fixed reference."""
        ids = np.asarray(ids)
        S = self.forward(ids[:-1])
        S = S - S.max(-1, keepdims=True)
        logp = S - np.log(np.exp(S).sum(-1, keepdims=True))
        return float(-logp[np.arange(len(ids) - 1), ids[1:]].mean())

    # ---- saving and loading -------------------------------------------------------------------
    def tables(self):
        t = {"E": self.E.table, "P": self.P.table}
        for l, b in enumerate(self.blocks):
            for name, arr in b.tables().items():
                t[f"h.{l}.{name}"] = arr
        if self.ln_f is not None:
            t["ln_f.gain"] = self.ln_f.gain
            if hasattr(self.ln_f, "bias"):
                t["ln_f.bias"] = self.ln_f.bias
        if self.W_out is not None:
            t["W_out"] = self.W_out
        return t

    def save(self, path):
        np.savez(path, **self.tables())

    def load(self, path_or_dict, check_shapes=True):
        d = dict(np.load(path_or_dict)) if isinstance(path_or_dict, str) else path_or_dict
        per_layer = {}
        for k, v in d.items():
            if k.startswith("h."):
                _, l, name = k.split(".", 2)
                per_layer.setdefault(int(l), {})[name] = v
            elif k == "E":
                self.E.table = v
            elif k == "P":
                self.P.table = v
            elif k == "ln_f.gain":
                self.ln_f.gain = v
            elif k == "ln_f.bias":
                self.ln_f.bias = v
            elif k == "W_out":
                self.W_out = v
        for l, tables in per_layer.items():
            self.blocks[l].set_tables(tables, check_shapes=check_shapes)
        return self
