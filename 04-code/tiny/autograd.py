"""A thirty-line autograd for NumPy (Chapter 18).  The class `T` is a table of numbers that remembers how it
was made: which tables went in (`parents`) and a rule (`back`) that turns the nudge arriving at the result into
the nudges for each parent.  `backward()` visits the record from the loss back to the inputs, applying each
rule once, and ADDS what arrives at each table, because a table used in several places collects a nudge from
every use (Chapter 17's merging paths).  Every rule is one of Chapter 17's slopes or Chapter 18's table rules:
a sum passes the nudge through, a product multiplies it by the other factor, Y = X W sends dY W^T to X and
X^T dY to W, e^x multiplies by e^x, ln x divides by x.  The class, from `class T` to the end of `backward`, is
28 lines; `unbroadcast`, `lift` and `scatter` are its helpers.  tiny_loss() below writes Tiny's forward pass with
it, and tiny.train.numpy_grads() uses it to give the same nudges as PyTorch, with nothing but NumPy."""
import numpy as np


def unbroadcast(g, shape):
    """NumPy stretched a smaller table to fit (broadcasting); its nudge is the sum over the stretched axes."""
    while g.ndim > len(shape):
        g = g.sum(0)
    return g.sum(tuple(i for i, s in enumerate(shape) if s == 1 and g.shape[i] != 1), keepdims=True)


def lift(x):
    return x if isinstance(x, T) else T(x)


def scatter(shape, i, g):
    """The nudge for a table read by index i: add each read's nudge back into the row it came from."""
    z = np.zeros(shape); np.add.at(z, i, g); return z


def mean(x, axis=-1):
    return x.sum(axis) * (1.0 / x.data.shape[axis])


class T:
    def __init__(s, data, parents=(), back=None):
        s.data, s.parents, s.back = np.asarray(data, dtype=float), parents, back
        s.grad = np.zeros_like(s.data)
    def __add__(s, o): o = lift(o); return T(s.data + o.data, (s, o), lambda g: (g, g))
    def __mul__(s, o): o = lift(o); return T(s.data * o.data, (s, o), lambda g: (g * o.data, g * s.data))
    def __matmul__(s, o): return T(s.data @ o.data, (s, o), lambda g: (g @ o.data.swapaxes(-1, -2), s.data.swapaxes(-1, -2) @ g))
    def __sub__(s, o): return s + lift(o) * -1.0
    def __truediv__(s, o): return s * lift(o) ** -1.0
    def __pow__(s, p): return T(s.data ** p, (s,), lambda g: (g * p * s.data ** (p - 1),))
    def exp(s): e = np.exp(s.data); return T(e, (s,), lambda g: (g * e,))
    def log(s): return T(np.log(s.data), (s,), lambda g: (g / s.data,))
    def tanh(s): t = np.tanh(s.data); return T(t, (s,), lambda g: (g * (1 - t * t),))
    def relu(s): return T(np.maximum(s.data, 0), (s,), lambda g: (g * (s.data > 0),))
    def sum(s, axis=-1): return T(s.data.sum(axis, keepdims=True), (s,), lambda g: (np.broadcast_to(g, s.data.shape),))
    def reshape(s, *shape): return T(s.data.reshape(*shape), (s,), lambda g: (g.reshape(s.data.shape),))
    def transpose(s, *ax): return T(s.data.transpose(*ax), (s,), lambda g: (g.transpose(np.argsort(ax)) if ax else g.T,))
    def __getitem__(s, i): return T(s.data[i], (s,), lambda g: (scatter(s.data.shape, i, g),))
    def backward(s):
        order, seen = [], set()
        def visit(t):
            if id(t) not in seen:
                seen.add(id(t)); [visit(p) for p in t.parents]; order.append(t)
        visit(s); s.grad = np.ones_like(s.data)
        for t in reversed(order):
            if t.back is not None:
                for p, g in zip(t.parents, t.back(t.grad)):
                    p.grad = p.grad + unbroadcast(np.asarray(g, dtype=float), p.data.shape)


# ---- Tiny's forward pass written with T (the same steps as tiny/model.py and tiny/torch_tiny.py) ----------
GELU_C = np.sqrt(2.0 / np.pi)


def softmax(x, mask=None):
    if mask is not None:
        x = x + np.where(mask, 0.0, -np.inf)                       # a constant: the mask is never nudged
    top = np.max(x.data, axis=-1, keepdims=True)                   # a constant too: softmax ignores a shift
    e = (x - top).exp()
    return e / e.sum(-1)


def norm(x, gain, bias=None, eps=1e-5):
    if bias is None:                                               # RMSNorm
        return x * (mean(x * x) + eps) ** -0.5 * gain
    c = x - mean(x)                                             # layer norm
    return c * (mean(c * c) + eps) ** -0.5 * gain + bias


def tiny_loss(t, ids, n_layers, n_heads, act="gelu", tied=True, final_norm=True, keep=None):
    """t: {table name: T}.  Returns the loss as a T; `keep`, a dict, collects named intermediates if given."""
    ids = np.asarray(ids); n = len(ids) - 1
    x = t["E"][ids[:-1]] + t["P"][np.arange(n)]
    mask = np.tril(np.ones((n, n), dtype=bool))
    for l in range(n_layers):
        p = f"h.{l}."
        a = norm(x, t[p + "ln_1.gain"], t.get(p + "ln_1.bias"))
        d = a.data.shape[-1] // n_heads
        split = lambda Z: Z.reshape(n, n_heads, d).transpose(1, 0, 2)
        Q, K, V = split(a @ t[p + "att.W_q"]), split(a @ t[p + "att.W_k"]), split(a @ t[p + "att.W_v"])
        P = softmax(Q @ K.transpose(0, 2, 1) * (1.0 / np.sqrt(d)), mask)
        O = (P @ V).transpose(1, 0, 2).reshape(n, n_heads * d)
        y = O @ t[p + "att.W_o"]
        if keep is not None and l == 0:
            keep.update(a=a, Q=Q, K=K, V=V, P=P, O=O, y=y)
        x = x + y
        f = norm(x, t[p + "ln_2.gain"], t.get(p + "ln_2.bias"))
        u = f @ t[p + "ff.W_1"] + t[p + "ff.b_1"]
        u = u.relu() if act == "relu" else u * ((u + u * u * u * 0.044715) * GELU_C).tanh() * 0.5 + u * 0.5
        x = x + u @ t[p + "ff.W_2"] + t[p + "ff.b_2"]
    if final_norm:
        x = norm(x, t["ln_f.gain"], t.get("ln_f.bias"))
    S = x @ (t["E"].transpose(1, 0) if tied else t["W_out"])
    logp = S - softmax_log_denominator(S)
    return mean(logp[np.arange(n), ids[1:]], 0) * -1.0


def softmax_log_denominator(S):
    top = np.max(S.data, axis=-1, keepdims=True)
    return (S - top).exp().sum(-1).log() + top


def grads(m, ids):
    """Loss and nudges for every table of NumPy Tiny m, by this file's autograd.  Same keys as Tiny.tables()."""
    tables = m.tables()
    t = {k: T(v.copy()) for k, v in tables.items()}
    b0 = m.blocks[0]
    L = tiny_loss(t, ids, m.n_layers, b0.att.n_heads, act=b0.ff.act_name, tied=m.tied, final_norm=m.ln_f is not None)
    L.backward()
    return float(L.data.ravel()[0]), {k: v.grad for k, v in t.items()}
