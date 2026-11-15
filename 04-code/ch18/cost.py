"""Chapter 18, section 18.10: what the backward pass costs.  One causal head, 1,024 seats, 32 wide (Tiny's head
size), seed 18, 64-bit NumPy, one thread, fastest of five runs, 10 November 2026 on the book's build machine.
Dense: the forward pass keeps the 1,024 x 1,024 share table and the backward pass reads it.  Tiled: the forward
pass keeps only O and each row's log-sum-exp, and tiny.attention.tiled_head_backward recomputes the shares in
64 x 64 blocks (FlashAttention's plan, in NumPy).  Both are checked against PyTorch autograd."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import _common
import numpy as np, torch
torch.set_num_threads(1)
from tiny.attention import softmax_backward, tiled_head_backward
from _common import best_of, save_json
rng = np.random.default_rng(18)
n, d = 1024, 32
Q, K, V, dO = (rng.standard_normal((n, d)) for _ in range(4))
c = 1 / np.sqrt(d)
mask = np.tril(np.ones((n, n), dtype=bool))
def forward():
    S = np.where(mask, Q @ K.T * c, -np.inf)
    mx = S.max(-1, keepdims=True); e = np.exp(S - mx); z = e.sum(-1, keepdims=True)
    P = e / z
    return P, P @ V, (mx + np.log(z)).ravel()
def backward(P):
    dP = dO @ V.T; dV = P.T @ dO
    dS = softmax_backward(P, dP)
    return dS @ K * c, dS.T @ Q * c, dV
P, O, lse = forward()
dQ, dK, dV = backward(P)
tQ, tK, tV = tiled_head_backward(Q, K, V, O, lse, dO)
q, k, v = (torch.tensor(x, requires_grad=True) for x in (Q, K, V))
s = (q @ k.T * c).masked_fill(~torch.tensor(mask), float("-inf"))
o = torch.softmax(s, -1) @ v; o.backward(torch.tensor(dO))
ref = (q.grad.numpy(), k.grad.numpy(), v.grad.numpy())
e_dense = max(np.abs(a - b).max() for a, b in zip((dQ, dK, dV), ref))
e_tiled = max(np.abs(a - b).max() for a, b in zip((tQ, tK, tV), ref))
print(f"one head, {n} seats, {d} wide: dense backward = autograd to {e_dense:.1e}; tiled = autograd to {e_tiled:.1e}")
tf = best_of(forward); tb = best_of(lambda: backward(P)); tt = best_of(lambda: tiled_head_backward(Q, K, V, O, lse, dO))
fl = n * n * d
print(f"products of size n x n x d (each {fl:,} multiply-adds, before the mask halves them):")
print("  forward: Q K^T and P V                         2")
print("  backward: dO V^T, P^T dO, dS K, dS^T Q          4")
print("  tiled backward: the same four, plus Q K^T again 5")
dS = softmax_backward(P, dO @ V.T)
pf = best_of(lambda: (Q @ K.T, P @ V)); pb = best_of(lambda: (dO @ V.T, P.T @ dO, dS @ K, dS.T @ Q))
print(f"time for the products alone: forward {pf:.1f} ms, backward {pb:.1f} ms ({pb/pf:.1f} x)")
print(f"time: forward {tf:.1f} ms; dense backward {tb:.1f} ms ({tb/tf:.1f} x forward); tiled backward {tt:.1f} ms ({tt/tf:.1f} x)")
kept_dense = P.nbytes + O.nbytes; kept_tiled = O.nbytes + lse.nbytes
print(f"kept between the passes (beyond Q, K, V): dense {kept_dense/1e6:.2f} MB (the share table and O); tiled {kept_tiled/1e6:.3f} MB (O and lse)")
save_json("cost", dict(pf=pf, pb=pb, e_dense=float(e_dense), e_tiled=float(e_tiled), tf=tf, tb=tb, tt=tt, kept_dense=kept_dense, kept_tiled=kept_tiled))
