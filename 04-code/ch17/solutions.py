"""Worked answers to Chapter 17's exercises (Appendix A).  64-bit; nothing random."""
import _common
import numpy as np
import torch
from hand import hand_tiny, IDS
from tiny.torch_tiny import TorchTiny

def wig(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

print("17.1 slopes at x = 2: 5x - 3 wiggled", round(wig(lambda x: 5 * x - 3, 2.0), 6), "rule 5;",
      "x cubed wiggled", round(wig(lambda x: x ** 3, 2.0), 6), "rule 3 x 2^2 = 12")
print("17.2 (3x + 1) cubed at x = 1: wiggled", round(wig(lambda x: (3 * x + 1) ** 3, 1.0), 4),
      "; chain rule 3 x 4^2 (outer) x 3 (inner) =", 3 * 4 ** 2 * 3)

m = hand_tiny(); S, st = m.forward(IDS, keep=True); TOP = m.top_rows(st[-1]); E0 = m.E.table.copy()
def loss_of_x(x):
    e = E0.copy(); e[4, 0] = x; z = TOP[:5] @ e.T; z = z - z.max(1, keepdims=True)
    return float(-(z - np.log(np.exp(z).sum(1, keepdims=True)))[np.arange(5), IDS[1:]].mean())
g = wig(loss_of_x, -1.0)
x1 = -1.0 - 4 * g
print(f"17.3 one step at lr 4 from x = -1: slope {g:+.5f}, x -> {x1:.4f}, loss {loss_of_x(-1.0):.4f} -> {loss_of_x(x1):.4f}")

x, t = 2.0, 0.5
w1, b1, w2, b2 = 0.5, -0.5, 1.5, 0.0
z = w1 * x + b1; h = max(0, z); y = w2 * h + b2
dy = 2 * (y - t); dz = dy * w2 * (z > 0)
print(f"17.4 target 0.5: y {y}, loss {(y - t) ** 2}, nudges w1 {dz * x}, b1 {dz}, w2 {dy * h}, b2 {dy}")
N = 2766592
print(f"17.5 Tiny, one pass over the held-out text: 6 x {N:,} x 55,860 = {6 * N * 55860:.3e} FLOPs;"
      f" one billion numbers on 20 billion tokens: 6 x 1e9 x 2e10 = {6 * 1e9 * 2e10:.1e} FLOPs")
lrs = np.linspace(0.01, 40, 4000)
ok = [lr for lr in lrs if loss_of_x(-1.0 - lr * g) <= loss_of_x(-1.0)]
print(f"17.6 critical learning rate from x = -1 (largest step that does not raise the loss): {max(ok):.2f};"
      f" 2 / bend at the bottom is 8.53")
tt = TorchTiny.from_numpy(m); L, G = tt.grads(IDS)
T = m.tables()["P"]; old = T[2, 1]
T[2, 1] = old + 1e-6; up = m.loss(IDS); T[2, 1] = old - 1e-6; dn = m.loss(IDS); T[2, 1] = old
print(f"17.7 hand Tiny, P[2][1]: autograd {G['P'][2, 1]:+.6f}, wiggled {(up - dn) / 2e-6:+.6f}")
