"""Chapter 8: softmax against four alternatives on sat's scaled, masked scores, and what a nudge does.
Run from 04-code/ch08:  python3 alternatives.py"""
from _common import *
from tiny.attention import softmax

s = np.array([1.414, 0.707, 2.121])                 # sat's scaled scores for the, cat, sat
def hardmax(x):  z = np.zeros_like(x); z[np.argmax(x)] = 1.0; return z
def sigmoid(x):  return 1 / (1 + np.exp(-x))
def relu_norm(x): r = np.maximum(x, 0); return r / r.sum() if r.sum() > 0 else r
def sparsemax(x):
    z = np.sort(x)[::-1]; k = np.arange(1, len(x) + 1); cs = np.cumsum(z)
    kk = k[1 + k * z > cs].max(); tau = (cs[kk - 1] - 1) / kk
    return np.maximum(x - tau, 0)
fns = [("softmax", softmax), ("hardmax", hardmax), ("sigmoid, each on its own", sigmoid),
       ("negatives to zero, then divide", relu_norm), ("sparsemax", sparsemax)]
V = (X @ W_v)[:3]
print("sat's scaled scores for the, cat, sat:", s.tolist())
for name, f in fns:
    w = f(s); w2 = f(s + np.array([0, 0.1, 0]))      # nudge cat's score up by 0.1
    print(f"  {name:<31} weights {np.round(w, 3).tolist()}  sum {w.sum():.3f}  output {tuple(np.round(w @ V, 3).tolist())}"
          f"  | cat's weight after a 0.1 nudge moves by {w2[1] - w[1]:+.4f}")
print("\nall scores very negative (-6, -6.1, -5.9):")
low = np.array([-6.0, -6.1, -5.9])
for name, f in fns:
    w = f(low); print(f"  {name:<31} weights {np.round(w, 4).tolist()}  sum {w.sum():.4f}")
