"""Tiny's layer as one object (Chapter 14).  A pre-norm transformer layer, the arrangement of
Chapter 13 and of GPT-2:

    x <- x + MultiHead(ln_1(x))        attention reads a normalised copy; its output is added
    x <- x + FeedForward(ln_2(x))      the feed-forward block does the same

The six steps (norm, attention, add, norm, feed-forward, add) are exposed by `explode`, which
returns every intermediate table, so the book can print one layer step by step.

Seeds: layer l uses seed 1865 + 100 * l for both its attention and feed-forward tables (the same
rule as Chapter 13's stacks), so layer 0 is exactly Chapter 13's layer.  Norm gains start at one and
biases at zero.

Chapter 15 adds three switches so that a Tiny small enough to trace by hand can be built from the
same classes: d_ff (the feed-forward width), activation (any name in tiny.ffn.ACTIVATIONS) and
norm ("layer" for LayerNorm, the default, or "rms" for RMSNorm, which has a gain and no bias).
With the defaults every table, seed and output is exactly Chapter 14's."""
import numpy as np
from tiny.attention import MultiHead
from tiny.ffn import FeedForward
from tiny.norm import LayerNorm, RMSNorm

SEED = 1865
LAYER_SEED_STEP = 100
NORMS = {"layer": LayerNorm, "rms": RMSNorm}


class Block:
    def __init__(self, layer=0, d_model=128, n_heads=4, d_ff=None, seed=SEED, spread=0.02,
                 activation="gelu", norm="layer"):
        s = seed + LAYER_SEED_STEP * layer
        self.layer = layer
        self.ln_1 = NORMS[norm](d_model)
        self.att = MultiHead(d_model=d_model, n_heads=n_heads, seed=s, spread=spread)
        self.ln_2 = NORMS[norm](d_model)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff, activation=activation, seed=s, spread=spread)

    @property
    def n_params(self):
        return self.ln_1.n_params + self.att.n_params + self.ln_2.n_params + self.ff.n_params

    def tables(self):
        """Every learned table in the layer, by name (the names `Tiny.save` uses)."""
        t = {"ln_1.gain": self.ln_1.gain}
        if hasattr(self.ln_1, "bias"):
            t["ln_1.bias"] = self.ln_1.bias
        t.update({"att.W_q": self.att.W_q, "att.W_k": self.att.W_k, "att.W_v": self.att.W_v, "att.W_o": self.att.W_o,
                  "ln_2.gain": self.ln_2.gain})
        if hasattr(self.ln_2, "bias"):
            t["ln_2.bias"] = self.ln_2.bias
        t.update({"ff.W_1": self.ff.W_1, "ff.b_1": self.ff.b_1, "ff.W_2": self.ff.W_2, "ff.b_2": self.ff.b_2})
        return t

    def set_tables(self, tables, check_shapes=True):
        """Load tables by name.  With check_shapes=True a table of the wrong shape is refused,
        even a square one that would run (Chapter 14, Lab 14, Breakage 1)."""
        owners = {"ln_1": self.ln_1, "ln_2": self.ln_2, "att": self.att, "ff": self.ff}
        for name, value in tables.items():
            part, attr = name.split(".")
            obj = owners[part]
            attr = {"gain": "gain", "bias": "bias"}.get(attr, attr)
            old = getattr(obj, attr)
            value = np.asarray(value)
            if check_shapes and value.shape != old.shape:
                raise ValueError(f"layer {self.layer} {name}: expected shape {old.shape}, got {value.shape}")
            setattr(obj, attr, value)

    def explode(self, X):
        """One layer in its six steps.  Returns a dict of every intermediate (n, d) table."""
        X = np.asarray(X, dtype=float)
        a_in = self.ln_1(X)
        a_out = self.att(a_in)
        A = X + a_out
        f_in = self.ln_2(A)
        f_out = self.ff(f_in)
        Y = A + f_out
        return {"x": X, "ln_1": a_in, "attention": a_out, "add_1": A, "ln_2": f_in, "feed_forward": f_out, "add_2": Y}

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        A = X + self.att(self.ln_1(X))
        return A + self.ff(self.ln_2(A))
