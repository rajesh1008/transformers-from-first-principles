"""Training Tiny (Chapter 17): nudging learned numbers downhill on the loss.

Three ways to get the nudges, from slowest to fastest, all returning {table name: array} keyed by the
names `Tiny.tables()` uses:

  wiggle_grads(m, ids, names)   NumPy only.  For each learned number, add a small amount h, measure the
                                loss, subtract h, measure again: slope ~ (L(+h) - L(-h)) / 2h.  Two
                                forward passes per number, so 2 x 112 passes for the hand Tiny and
                                2 x 2,766,592 for Tiny.  It needs no calculus and it is the check.
  torch_grads(m, ids)           PyTorch autograd on a copy of the same tables (tiny/torch_tiny.py):
                                one forward and one backward pass for all numbers at once.
  numpy_grads(m, ids)           Chapter 18's thirty-line NumPy autograd (tiny/autograd.py): the same one
                                forward and one backward pass, with every rule written out in NumPy.

and one rule for using them, plain gradient descent:

  sgd_step(m, grads, lr)        every number moves by -lr x its nudge.

`train(m, batches, steps, lr, grad_fn)` repeats step and records the loss; it works with any
gradient function and leaves the NumPy Tiny holding the trained tables.  Chapter 20 replaces
sgd_step with momentum, Adam and schedules; Chapter 24 trains Tiny for real with them."""
import numpy as np


def wiggle_grads(m, ids, names=None, h=1e-6, loss_fn=None):
    """Central-difference nudge for every number of the named tables (default: all of them)."""
    loss_fn = loss_fn or (lambda: m.loss(ids))
    tables = m.tables()
    names = list(tables) if names is None else names
    out = {}
    for name in names:
        T = tables[name]
        g = np.zeros_like(T, dtype=float)
        it = np.nditer(T, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = T[i]
            T[i] = old + h; up = loss_fn()
            T[i] = old - h; down = loss_fn()
            T[i] = old
            g[i] = (up - down) / (2 * h)
        out[name] = g
    return out


def numpy_grads(m, ids):
    """Loss and nudges from tiny/autograd.py; agrees with torch_grads to about 1e-15 in 64-bit."""
    from tiny.autograd import grads
    return grads(m, ids)


def torch_grads(m, ids, dtype=None):
    """Loss and autograd nudges for every table of NumPy Tiny m (64-bit unless dtype is given)."""
    import torch
    from tiny.torch_tiny import TorchTiny
    tt = TorchTiny.from_numpy(m, dtype=dtype or torch.float64)
    return tt.grads(ids)


def sgd_step(m, grads, lr):
    """Plain gradient descent: every learned number moves by -lr times its nudge, in place."""
    tables = m.tables()
    for name, g in grads.items():
        tables[name] -= lr * g          # the arrays in m.tables() are m's own tables, so m changes
    return m


def train(m, batches, steps, lr, grad_fn="torch", every=1):
    """Gradient descent for `steps` steps.  batches(step) returns the ids for that step.
    grad_fn "torch" (autograd), "numpy" (Chapter 18's autograd) or "wiggle" (finite differences).  Returns [(step, loss)] recorded
    every `every` steps, the loss being the one measured before that step's nudge."""
    history = []
    for s in range(steps):
        ids = batches(s)
        if grad_fn == "wiggle":
            L = m.loss(ids)
            g = wiggle_grads(m, ids)
        elif grad_fn == "numpy":
            L, g = numpy_grads(m, ids)
        else:
            L, g = torch_grads(m, ids)
        if s % every == 0:
            history.append((s, float(L)))
        sgd_step(m, g, lr)
    return history


def global_norm(grads):
    """Chapter 19: the length of all nudges taken together, sqrt(sum of every nudge squared), in 64-bit."""
    import torch
    return float(torch.sqrt(sum((g.detach().double() ** 2).sum() for g in grads)))


def train_torch(tt, batches, steps, lr, every=1, on_step=None, clip=None, norms=None):
    """The same plain gradient descent on a TorchTiny directly (no copy back to NumPy each step),
    for runs on the novel.  Returns [(step, loss)]; a loss that is not finite is recorded and the run
    stops there, because every later number would be nan.
    Chapter 19: clip=c rescales all nudges together whenever their global norm exceeds c (gradient
    clipping by global norm, the norm taken in 64-bit); norms, if a list, collects each step's global
    norm before clipping.  With clip=None and norms=None this is Chapter 17's loop unchanged."""
    import torch
    history = []
    for s in range(steps):
        for v in tt.t.values():
            v.grad = None
        L = tt.loss(batches(s))
        L.backward()
        if s % every == 0 or not torch.isfinite(L):
            history.append((s, float(L.detach())))
        if not torch.isfinite(L):
            break
        factor = 1.0
        if clip is not None or norms is not None:
            gn = global_norm([v.grad for v in tt.t.values()])
            if norms is not None:
                norms.append(gn)
            if clip is not None and gn > clip:
                factor = clip / gn
        with torch.no_grad():
            for v in tt.t.values():
                v -= lr * factor * v.grad
        if on_step:
            on_step(s, float(L.detach()))
    return history
