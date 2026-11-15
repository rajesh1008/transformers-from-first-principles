"""Choosing Tiny's starting numbers (Chapter 19).

Every learned table of Tiny was drawn by Chapters 4 to 14 as  spread x (a standard normal draw), with its own
seed: spread 0.02 for the token table and for every table inside a layer, 0.01 for the position table (GPT-2's
choices).  `initialise` keeps those same standard normal draws and only changes the spread, so two schemes
differ in nothing but the size of their numbers, and any difference you measure is caused by size alone.

Schemes (the layers' tables are W_q, W_k, W_v, W_o in attention and W_1, W_2 in the feed-forward block;
biases stay 0, norm gains 1, norm biases 0):

  "gpt2"           0.02 everywhere (Tiny's default; initialise(m, "gpt2") changes nothing)
  "scaled"         as gpt2, but the two tables that write into the residual stream, W_o and W_2, get
                   0.02 / sqrt(2 x n_layers)  (the residual scaling of GPT-2's paper and Megatron-LM)
  "fan_in"         every layer table gets 1 / sqrt(fan_in), fan_in = the number of rows it reads
                   (128 for W_q, W_k, W_v, W_o and W_1; 512 for W_2)
  "fan_in_scaled"  fan_in, with W_o and W_2 also divided by sqrt(2 x n_layers)
  "zeros"          every layer table 0 (token and position tables kept)
  "all_zeros"      every learned number 0, token and position tables included
  "constant"       every layer table filled with the single number 0.02 (no randomness)

`scale` multiplies every layer table after the scheme is applied (scale=10 is "ten times too large").
The token and position tables are left as drawn unless the scheme is "all_zeros"; `embed_scale` multiplies
them if you want to change them too."""
import numpy as np

LAYER_TABLES = ("att.W_q", "att.W_k", "att.W_v", "att.W_o", "ff.W_1", "ff.W_2")
RESIDUAL_WRITERS = ("att.W_o", "ff.W_2")
SCHEMES = ("gpt2", "scaled", "fan_in", "fan_in_scaled", "zeros", "all_zeros", "constant")
DRAWN_SPREAD = {"E": 0.02, "P": 0.01}        # what the token and position tables were drawn with


def target_spread(name, shape, scheme, n_layers):
    """The spread a layer table should have under a scheme (name like 'att.W_o', shape of the table)."""
    if scheme in ("gpt2", "scaled", "constant"):
        s = 0.02
    elif scheme in ("fan_in", "fan_in_scaled"):
        s = 1.0 / np.sqrt(shape[0])
    elif scheme in ("zeros", "all_zeros"):
        return 0.0
    else:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")
    if scheme in ("scaled", "fan_in_scaled") and name in RESIDUAL_WRITERS:
        s = s / np.sqrt(2 * n_layers)
    return s


def initialise(m, scheme="gpt2", scale=1.0, embed_scale=1.0):
    """Re-size every table of Tiny m in place and return m.  The random draws are the ones Tiny already
    holds (each table divided by the spread it was drawn with), so a scheme changes sizes, never the dice."""
    L = m.n_layers
    for b in m.blocks:
        for name in LAYER_TABLES:
            part, attr = name.split(".")
            obj = b.att if part == "att" else b.ff
            T = getattr(obj, attr)
            if not hasattr(obj, "_draws"):
                obj._draws = {}
            z = obj._draws.setdefault(attr, T / 0.02)           # the standard normal draws, kept once
            s = target_spread(name, T.shape, scheme, L)
            if scheme == "constant":
                new = np.full(T.shape, s)
            else:
                new = z * s
            setattr(obj, attr, new * scale)
    for key, tab in (("E", m.E), ("P", m.P)):
        if not hasattr(tab, "_draws"):
            tab._draws = tab.table / DRAWN_SPREAD[key]
        tab.table = tab._draws * DRAWN_SPREAD[key] * (0.0 if scheme == "all_zeros" else embed_scale)
    m.init_scheme = (scheme, scale)
    return m


def spreads(m):
    """The spread (root mean square) of every table of layer 0, plus the token and position tables."""
    out = {"E": float(np.sqrt((m.E.table ** 2).mean())), "P": float(np.sqrt((m.P.table ** 2).mean()))}
    b = m.blocks[0]
    for name in LAYER_TABLES:
        part, attr = name.split(".")
        T = getattr(b.att if part == "att" else b.ff, attr)
        out[name] = float(np.sqrt((T ** 2).mean()))
    return out
