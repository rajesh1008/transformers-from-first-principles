"""Tiny's normalisation (Chapter 13).  Each row is rescaled on its own, using only its own numbers:
no row sees any other row, as with the feed-forward block of Chapter 12.

LayerNorm: subtract the row's average, divide by its spread (the square root of the average squared
distance from the average, with a small eps inside the root so a flat row does not divide by zero),
then multiply by a learned gain and add a learned bias, one of each per column.  GPT-2 uses this.
RMSNorm: skip the average; divide by the root of the average square; multiply by a gain.  No bias.
Most models published since 2023 use this.

Population spread (divide by d, not d - 1), eps = 1e-5 inside the root: the same as GPT-2 and PyTorch.
Gains start at one and biases at zero, so an untrained norm does nothing but the rescaling.

Also here: token_similarity, the share of a table's total squared size that lies in its average row
(1 when every row is the same, near 0 when rows are unrelated), and layer_step, the four ways the
book compares of wiring a block, its residual and a norm together."""
import numpy as np

EPS = 1e-5


def average(x):
    return np.asarray(x, dtype=float).mean(-1, keepdims=True)


def spread(x):
    """Population standard deviation of each row: the typical distance from the row's average."""
    x = np.asarray(x, dtype=float)
    return np.sqrt(((x - average(x)) ** 2).mean(-1, keepdims=True))


def rms(x):
    """Root of the average square of each row."""
    x = np.asarray(x, dtype=float)
    return np.sqrt((x ** 2).mean(-1, keepdims=True))


def layer_norm(x, gain=None, bias=None, eps=EPS):
    x = np.asarray(x, dtype=float)
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    y = (x - m) / np.sqrt(v + eps)
    if gain is not None:
        y = y * gain
    if bias is not None:
        y = y + bias
    return y


def rms_norm(x, gain=None, eps=EPS):
    x = np.asarray(x, dtype=float)
    y = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)
    return y * gain if gain is not None else y


class LayerNorm:
    def __init__(self, d_model=128, eps=EPS):
        self.d_model, self.eps = d_model, eps
        self.gain = np.ones(d_model)
        self.bias = np.zeros(d_model)

    @property
    def n_params(self):
        return self.gain.size + self.bias.size

    def __call__(self, X):
        return layer_norm(X, self.gain, self.bias, self.eps)


class RMSNorm:
    def __init__(self, d_model=128, eps=EPS):
        self.d_model, self.eps = d_model, eps
        self.gain = np.ones(d_model)

    @property
    def n_params(self):
        return self.gain.size

    def __call__(self, X):
        return rms_norm(X, self.gain, self.eps)


def token_similarity(X):
    """||mean row||^2 * n / ||X||^2: the share of the table's squared size in what all rows share."""
    X = np.asarray(X, dtype=float)
    m = X.mean(0, keepdims=True)
    return float((m ** 2).sum() * len(X) / (X ** 2).sum())


def layer_step(x, block, norm, wiring):
    """One block (attention or feed-forward) applied with one of the wirings compared in Chapter 13.
      "none"     : x <- block(x)                 (no residual, no norm: the block replaces the row)
      "residual" : x <- x + block(x)             (Chapter 12's Tiny)
      "post"     : x <- norm(x + block(x))       (the 2017 paper's order, also BERT's)
      "pre"      : x <- x + block(norm(x))       (GPT-2 and nearly every model since)"""
    if wiring == "none":
        return block(x)
    if wiring == "residual":
        return x + block(x)
    if wiring == "post":
        return norm(x + block(x))
    if wiring == "pre":
        return x + block(norm(x))
    raise ValueError(wiring)
