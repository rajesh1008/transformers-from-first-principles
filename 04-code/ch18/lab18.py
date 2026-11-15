"""Chapter 18 lab: one head of the hand Tiny backwards three ways (the chapter's rules by hand, the thirty-line
autograd, PyTorch), four ways to break the hand version, and seven checkpoint tests.  64-bit; nothing random.
Run:  python lab18.py     (exits with an error if a checkpoint fails)"""
import _common
import numpy as np
import tiny.autograd as ag
from hand import hand_tiny, IDS
from tiny.attention import head_forward, head_backward, softmax_backward
from tiny.train import torch_grads
from tiny.model import Tiny
from _common import traced, npy, novel_ids, row

m = hand_tiny(); b = m.blocks[0]
L, k = traced(m, IDS); L.backward()
dY = npy(k["Y"].grad).squeeze()
A = b.ln_1(m.embed(IDS[:-1]))
f = head_forward(A, b.att.W_q, b.att.W_k, b.att.W_v, b.att.W_o)
g = head_backward(f, dY)
torch_tab = {n: torch_grads(m, IDS)[1]["h.0.att." + n] for n in ("W_q", "W_k", "W_v")}

# 1. the same head through the thirty-line autograd: pretend loss = sum(Y x dY), whose nudge on Y is dY
t = {n: ag.T(getattr(b.att, n).copy()) for n in ("W_q", "W_k", "W_v", "W_o")}
tA = ag.T(A.copy())
Qt, Kt, Vt = tA @ t["W_q"], tA @ t["W_k"], tA @ t["W_v"]
Pt = ag.softmax(Qt @ Kt.transpose() * (1 / np.sqrt(2)), mask=np.tril(np.ones((5, 5), dtype=bool)))
Yt = (Pt @ Vt) @ t["W_o"]
(Yt * ag.T(dY)).sum().backward()
print("three ways to the nudge on W_q, W_k, W_v:")
for n in ("W_q", "W_k", "W_v"):
    print(f"  {n}: by hand {np.round(g[n], 5).tolist()}  thirty-line autograd differs by {np.abs(t[n].grad - g[n]).max():.0e}, "
          f"PyTorch by {np.abs(torch_tab[n] - g[n]).max():.0e}")

# 2. breakages
def broken(kind):
    c = f["c"]
    dP = dY @ f["V"].T
    dS = f["P"] * dP if kind == "no row-sum term" else softmax_backward(f["P"], dP)
    cc = 1.0 if kind == "forgot the scale" else c
    dQ = dS @ f["K"] * cc
    dK = (dS @ f["Q"] if kind == "dS not transposed" else dS.T @ f["Q"]) * cc
    dV = f["P"].T @ dY
    out = dict(W_q=A.T @ dQ, W_k=A.T @ dK, W_v=A.T @ dV, Q=dQ)
    dA = dQ @ f["W_q"].T if kind == "query path only" else dQ @ f["W_q"].T + dK @ f["W_k"].T + dV @ f["W_v"].T
    out["A"] = dA
    return out
print("\nbreak it (largest error against the true nudge; does the sat@2 checkpoint notice; does comparing the three\n"
      "tables with PyTorch, or the nudge on the rows A with PyTorch's, notice):")
for kind in ("no row-sum term", "dS not transposed", "forgot the scale", "query path only"):
    o = broken(kind)
    err = {n: float(np.abs(o[n] - g[n]).max()) for n in ("W_q", "W_k", "W_v", "A")}
    worst = max(err, key=err.get)
    caught = np.abs(o["Q"][2] - g["Q"][2]).max() > 1e-6
    by_table = max(np.abs(o[n] - torch_tab[n]).max() for n in torch_tab) > 1e-15
    by_rows = np.abs(o["A"] - npy(k["a"].grad)).max() > 1e-15
    print(f"  {kind:<18} W_q {err['W_q']:.1e}  W_k {err['W_k']:.1e}  W_v {err['W_v']:.1e}  A {err['A']:.1e}"
          f"  sat@2's query {row(o['Q'][2], 5)} {'caught' if caught else 'missed'}"
          f"; table check {'catches it' if by_table else 'misses it'}; row check {'catches it' if by_rows else 'misses it'}")

# 3. checkpoints
checks = []
def check(name, ok):
    checks.append(ok); print(f"  [{'pass' if ok else 'FAIL'}] {name}")
print("\ncheckpoints:")
check(f"sat@2's query nudge is (+0.00256, -0.00233) to five places: {row(g['Q'][2], 5)}",
      np.allclose(np.round(g["Q"][2], 5), [0.00256, -0.00233]))
check(f"every row of dS adds to zero (largest {np.abs(g['S'].sum(-1)).max():.0e})", np.abs(g["S"].sum(-1)).max() < 1e-15)
check("the@0's query nudge is exactly zero", bool((g["Q"][0] == 0).all()))
check("dW_q, dW_k, dW_v by hand match PyTorch to 1e-15", max(np.abs(torch_tab[n] - g[n]).max() for n in torch_tab) < 1e-15)
W = b.att.W_q; old = W[1, 0]; h = 1e-6
W[1, 0] = old + h; up = m.loss(IDS); W[1, 0] = old - h; dn = m.loss(IDS); W[1, 0] = old
check(f"wiggling W_q[1, 0] gives {(up - dn) / (2 * h):+.8f}, the rule {g['W_q'][1, 0]:+.8f}", abs((up - dn) / (2 * h) - g["W_q"][1, 0]) < 1e-8)
La, Ga = ag.grads(m, IDS); Lt, Gt = torch_grads(m, IDS)
check("the thirty-line autograd's 112 nudges match PyTorch to 1e-14", max(np.abs(Ga[n] - Gt[n]).max() for n in Gt) < 1e-14)
mt = Tiny(); lt = mt.loss(novel_ids()[:1025])
check(f"Tiny unchanged: {mt.n_params:,} learned numbers, loss {lt:.4f} on the first 1,024 predictions",
      mt.n_params == 2766592 and round(lt, 4) == 7.6334)
print(f"\n{sum(checks)} of {len(checks)} checkpoints pass")
assert all(checks)
