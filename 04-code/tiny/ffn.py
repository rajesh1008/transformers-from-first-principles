"""Tiny's feed-forward block (Chapter 12).  Each row goes through the block on its own: no row sees any
other row.  The block widens the row from d_model to d_ff numbers with the up table W_1 (plus a bias),
applies an activation to each number separately, and narrows it back with the down table W_2 (plus a
bias).  Read as a memory: column j of W_1 is a key, the activation is how well the row matches it,
and row j of W_2 is the value added to the row in proportion.  Each (key, value) pair is a neuron.

gated=True gives the GLU form used by most models since 2020: a second up table W_g whose activated
output multiplies W_1's output number by number, with no activation on W_1's side.  The book uses
d_ff = 4 * d_model for the plain block and, when gated, d_ff = 8/3 * d_model rounded to a multiple of 64
so the learned numbers stay about the same.  Seeds follow Chapter 9's pattern (seed + 20 onwards)."""
import math
import numpy as np


def relu(x):
    return np.maximum(x, 0.0)


def gelu(x):
    """GPT-2's tanh form of the Gaussian error linear unit."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


_erf = np.vectorize(math.erf, otypes=[float])


def gelu_exact(x):
    """x times the chance that a standard normal number is below x."""
    x = np.asarray(x, dtype=float)
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def silu(x):
    """Also called swish: x times the logistic sigmoid of x."""
    return x / (1.0 + np.exp(-x))


ACTIVATIONS = {"relu": relu, "gelu": gelu, "gelu_exact": gelu_exact, "silu": silu}


class FeedForward:
    def __init__(self, d_model=128, d_ff=None, activation="gelu", gated=False, seed=1865, spread=0.02):
        if d_ff is None:
            d_ff = 4 * d_model if not gated else int(round(8 * d_model / 3 / 64)) * 64
        self.d_model, self.d_ff, self.gated = d_model, d_ff, gated
        self.act_name = activation
        self.act = ACTIVATIONS[activation]
        self.W_1 = np.random.default_rng(seed + 20).normal(0.0, spread, size=(d_model, d_ff))
        self.b_1 = np.zeros(d_ff)
        self.W_2 = np.random.default_rng(seed + 21).normal(0.0, spread, size=(d_ff, d_model))
        self.b_2 = np.zeros(d_model)
        self.W_g = np.random.default_rng(seed + 22).normal(0.0, spread, size=(d_model, d_ff)) if gated else None

    @property
    def n_params(self):
        n = self.W_1.size + self.b_1.size + self.W_2.size + self.b_2.size
        return n + (self.W_g.size if self.gated else 0)

    def hidden(self, X):
        """The d_ff neuron activations for every row: (n, d_ff)."""
        X = np.asarray(X, dtype=float)
        up = X @ self.W_1 + self.b_1
        if self.gated:
            return self.act(X @ self.W_g) * up
        return self.act(up)

    def __call__(self, X):
        """The block's output for every row (what is added to the row): (n, d_model)."""
        return self.hidden(X) @ self.W_2 + self.b_2

    def keys(self):
        """Column j of W_1 as row j: the pattern neuron j responds to."""
        return self.W_1.T

    def values(self):
        """Row j of W_2: what neuron j adds, per unit of activation."""
        return self.W_2
