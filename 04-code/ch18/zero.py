"""Chapter 18, sidebar "The gradient that was exactly zero": nudges that are zero by construction, not by
rounding.  (1) Every masked score.  (2) The first seat's query: its row of shares is a single 1, whose slope in
its own score is 1 x (1 - 1) = 0.  (3) The sixth seat, mat@5, when the head is run on all six seats as in
Chapter 15: no loss reads it, so its row of dY is zero and so are its query, key and value nudges.  (4) A second
setting: the real Tiny, all twelve layers and four heads, on the novel's first 64 tokens.  64-bit."""
import _common
import numpy as np, torch
from hand import hand_tiny, IDS
from tiny.attention import head_forward, head_backward
from _common import load_json, traced, npy, novel_ids
j = load_json("backward_head")
g = {k: np.array(v) for k, v in j["g"].items()}
n = 5
masked = ~np.tril(np.ones((n, n), dtype=bool))
print(f"(1) masked scores: {masked.sum()} of {n*n}; their nudges are all exactly zero: {bool((g['S'][masked] == 0).all())}")
print(f"(2) the@0's query nudge: {g['Q'][0].tolist()}  (its share row is [1, 0, 0, 0, 0])")
print(f"    the@0's whole row of dS: {[float(v) + 0.0 for v in g['S'][0]]}")
m = hand_tiny(); b = m.blocks[0]
X6 = m.embed(IDS); A6 = b.ln_1(X6)
f6 = head_forward(A6, b.att.W_q, b.att.W_k, b.att.W_v, b.att.W_o)
dY6 = np.vstack([np.array(j["dY"]), np.zeros((1, 2))])
g6 = head_backward(f6, dY6)
print(f"(3) six seats, mat@5's row of dY set to zero (nothing reads it): mat@5's dQ {g6['Q'][5].tolist()}, dK {g6['K'][5].tolist()}, dV {g6['V'][5].tolist()}")
print(f"    the five-seat nudges are unchanged: largest difference in dW_q, dW_k, dW_v "
      f"{max(np.abs(g6[k] - g[k]).max() for k in ('W_q', 'W_k', 'W_v')):.1e}")
from tiny.model import Tiny
mt = Tiny(); ids = novel_ids()[:65]
zeros, total = 0, 0
for layer in range(mt.n_layers):
    L, k = traced(mt, ids, layer=layer); L.backward()
    dQ = npy(k["Q"].grad)                               # (heads, seats, d_head)
    zeros += int((dQ[:, 0, :] == 0).all(axis=-1).sum()); total += dQ.shape[0]
    if layer == 0:
        other = np.abs(dQ[:, 1:, :]).max()
print(f"(4) Tiny, 64 tokens: seat 0's query nudge is exactly zero in {zeros} of {total} heads (12 layers x 4 heads);")
print(f"    the largest query nudge at any other seat of layer 0 is {other:.2e}")
