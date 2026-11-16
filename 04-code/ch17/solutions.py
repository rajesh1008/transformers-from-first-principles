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

# Exercises 17.8 to 17.12 (added 16 November 2026).
import math
def share_slope(s, rest):
    p = math.exp(s) / (math.exp(s) + rest); return p, wig(lambda v: -math.log(math.exp(v) / (math.exp(v) + rest)), s)
for s in (0.5, 2.5):
    p, sl = share_slope(s, 3.0)
    print(f"17.8 s = {s}, rest = 3.0: share {p:.4f}; slope of the surprise wiggled {sl:+.4f}; rule share - 1 = {p - 1:+.4f}")
c1 = 0.407; r = 1 / c1 ** 2 - 1
print(f"17.9 noise-to-signal at one window: 1/0.407^2 - 1 = {r:.3f}; predicted cosine at 2, 8, 32 windows:",
      ", ".join(f"{k}: {1 / math.sqrt(1 + r / k):.3f}" for k in (2, 8, 32)), "(measured 0.606, 0.816, 0.934)")
from tiny.train import train
mm = hand_tiny(); hist = train(mm, lambda s: IDS, 200, 1.0); Ls = [L for _, L in hist]
pk = [i for i in range(1, len(Ls) - 1) if Ls[i] > Ls[i - 1] and Ls[i] > Ls[i + 1] and Ls[i] > 1.5 * min(Ls[max(0, i - 5):i])]
print(f"17.10 hand Tiny at lr 1.0, 200 steps: loss at steps 0, 10, 50, 199: {Ls[0]:.4g} {Ls[10]:.4g} {Ls[50]:.4g} {Ls[-1]:.4g};"
      f" largest {max(Ls):.4g}; spikes (peak 1.5 x the lowest of the five steps before): {len(pk)}" + (f", first at step {pk[0]}" if pk else ""))
g3 = wig(loss_of_x, 3.0)
ok3 = [lr for lr in lrs if loss_of_x(3.0 - lr * g3) <= loss_of_x(3.0)]
print(f"17.12 from x = +3: loss {loss_of_x(3.0):.4f}, slope {g3:+.5f}; critical learning rate {max(ok3):.2f}; 2 / bend at the bottom 8.53")
