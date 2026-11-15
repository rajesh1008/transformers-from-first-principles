"""Optimisers and learning-rate schedules for Tiny (Chapter 20).

Chapter 17's loop moved every learned number by -lr x its nudge.  This module keeps that loop's shape and
swaps the one line that moves the numbers for an optimiser object:

  SGD(lr, momentum=0.0)                plain gradient descent, or with a running average of the nudges
  Adam(lr, b1, b2, eps, wd=0.0)        a running average of the nudges divided by the square root of a
                                       running average of their squares, per number, with bias correction;
                                       wd > 0 gives AdamW (decoupled weight decay on tables with two axes)
  Muon(lr, momentum, adam=Adam(...))   for the square-ish tables inside the layers, the momentum table is
                                       orthogonalised by five Newton-Schulz steps; every other table
                                       (E, P, gains, biases) goes to the Adam it is given

Schedules are functions of the step that return a multiplier for lr:
  constant(), warmup_cosine(warmup, total, floor), wsd(warmup, total, decay_frac), linear_decay(warmup, total)

train_opt(tt, batches, steps, opt, schedule, clip, norms, on_step) is tiny/train.py's train_torch with the
optimiser and schedule plugged in.  SGD(lr) with constant() and no clip is exactly train_torch(tt, ..., lr):
the Chapter 20 lab checks it number for number.  Seed 1865 is not needed here: nothing in this module is random."""
import math
import torch


def _decays(name, v):
    """Weight decay applies to tables with two axes (E, P, W_q, ..., W_2), not to gains and biases."""
    return v.ndim == 2


class SGD:
    def __init__(self, lr, momentum=0.0):
        self.lr, self.mu, self.state = lr, momentum, {}

    @torch.no_grad()
    def step(self, params, mult=1.0):
        for name, v in params.items():
            g = v.grad
            if self.mu:
                b = self.state.get(name)
                b = g.clone() if b is None else b.mul_(self.mu).add_(g)
                self.state[name] = b
                g = b
            v.sub_(self.lr * mult * g)


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.95, eps=1e-8, wd=0.0, bias_correction=True):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.bc, self.t, self.m, self.v = bias_correction, 0, {}, {}

    def update(self, name, g):
        """The step direction Adam takes for one table (before lr): m_hat / (sqrt(v_hat) + eps)."""
        m = self.m.get(name)
        if m is None:
            m = self.m[name] = torch.zeros_like(g)
            self.v[name] = torch.zeros_like(g)
        v = self.v[name]
        m.mul_(self.b1).add_(g, alpha=1 - self.b1)            # running average of the nudge
        v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)     # running average of its square
        if self.bc:
            mh, vh = m / (1 - self.b1 ** self.t), v / (1 - self.b2 ** self.t)
        else:
            mh, vh = m, v
        return mh / (vh.sqrt() + self.eps)

    @torch.no_grad()
    def step(self, params, mult=1.0, names=None):
        self.t += 1
        for name, v in params.items():
            if names is not None and name not in names:
                continue
            lr = self.lr * mult
            if self.wd and _decays(name, v):
                v.mul_(1 - lr * self.wd)                      # AdamW: shrink the number, separately
            v.sub_(lr * self.update(name, v.grad))


def newton_schulz(G, steps=5):
    """Approximately replace G by the nearest table with every singular value 1 (Muon's quintic iteration)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


class Muon:
    """Muon on the tables inside the layers; Adam (AdamW) on the rest.  The orthogonalised step is scaled by
    0.2 x sqrt(larger side) so that its size per number is about Adam's, the convention that lets one
    learning rate serve both."""
    def __init__(self, lr, momentum=0.95, wd=0.0, adam=None):
        self.lr, self.mu, self.wd, self.buf = lr, momentum, wd, {}
        self.adam = adam

    @staticmethod
    def takes(name, v):
        return v.ndim == 2 and name not in ("E", "P")

    @torch.no_grad()
    def step(self, params, mult=1.0):
        rest = [n for n, v in params.items() if not self.takes(n, v)]
        self.adam.step(params, mult, names=set(rest))
        lr = self.lr * mult
        for name, v in params.items():
            if not self.takes(name, v):
                continue
            b = self.buf.get(name)
            b = v.grad.clone() if b is None else b.mul_(self.mu).add_(v.grad)
            self.buf[name] = b
            g = v.grad + self.mu * b                          # Nesterov-style look-ahead
            O = newton_schulz(g)
            if self.wd:
                v.mul_(1 - lr * self.wd)
            v.sub_(lr * 0.2 * math.sqrt(max(v.shape)) * O)


def constant():
    return lambda s: 1.0


def warmup_cosine(warmup, total, floor=0.1):
    def f(s):
        if s < warmup:
            return (s + 1) / warmup
        p = (s - warmup) / max(1, total - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))
    return f


def wsd(warmup, total, decay_frac=0.2, floor=0.0):
    """Warmup, then stable at the peak, then a straight-line decay over the last decay_frac of the steps."""
    d0 = int(round(total * (1 - decay_frac)))
    def f(s):
        if s < warmup:
            return (s + 1) / warmup
        if s < d0:
            return 1.0
        return floor + (1 - floor) * (total - s) / (total - d0)
    return f


def linear_decay(warmup, total, floor=0.0):
    def f(s):
        if s < warmup:
            return (s + 1) / warmup
        return floor + (1 - floor) * (total - s) / max(1, total - warmup)
    return f


def train_opt(tt, batches, steps, opt, schedule=None, clip=None, norms=None, on_step=None, start=0):
    """train_torch with an optimiser: returns [(step, loss)]; stops at a loss that is not finite."""
    from tiny.train import global_norm
    schedule = schedule or constant()
    history = []
    for s in range(start, start + steps):
        for v in tt.t.values():
            v.grad = None
        L = tt.loss(batches(s))
        L.backward()
        history.append((s, float(L.detach())))
        if not torch.isfinite(L):
            break
        if clip is not None or norms is not None:
            gn = global_norm([v.grad for v in tt.t.values()])
            if norms is not None:
                norms.append(gn)
            if clip is not None and gn > clip:
                with torch.no_grad():
                    for v in tt.t.values():
                        v.grad.mul_(clip / gn)
        opt.step(tt.t, schedule(s))
        if on_step:
            on_step(s, history[-1][1])
    return history
