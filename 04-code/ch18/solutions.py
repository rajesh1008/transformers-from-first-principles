"""Chapter 18: worked numbers for the exercises (Appendix A).  64-bit; seed 18 where anything is random."""
import _common, copy
import numpy as np, torch
from hand import hand_tiny, IDS
from tiny.attention import head_forward, head_backward, softmax
from tiny.train import torch_grads
from _common import load_json, traced, npy, row, f4
rc = load_json("recap"); j = load_json("backward_head")
P = np.array(rc["P"]); K = np.array(rc["K"]); Q = np.array(rc["Q"])
g = {k: np.array(v) for k, v in j["g"].items()}
c = 1 / np.sqrt(2)
print("E18.1  cat@1's two shares and their slope")
p = P[1, :2]; dP = g["P"][1, :2]
print(f"  shares {row(p)}; p(1 - p) = {p[0]*(1-p[0]):.4f}; dP {row(dP, 5)}")
print(f"  dS = p(1-p) x (dP0 - dP1) x (+1, -1) = {row(p[0]*(1-p[0])*(dP[0]-dP[1])*np.array([1,-1]), 5)}; chain {row(g['S'][1,:2], 5)}")
print("E18.3  on@3's query nudge")
print(f"  dS row {row(g['S'][3,:4], 5)}")
print(f"  dQ = (1/sqrt 2) sum dS_j K_j = {row(c * g['S'][3,:4] @ K[:4], 5)}; chain {row(g['Q'][3], 5)}")
print("E18.4  the@0's key nudge, from its five readers")
parts = [g["S"][i, 0] * Q[i] * c for i in range(5)]
for i, pp in enumerate(parts):
    print(f"  from seat {i}: {row(pp, 5)}")
print(f"  sum {row(sum(parts), 5)}; chain {row(g['K'][0], 5)}")
print("E18.5  the row-sum identity on sat@2: rowsum(P x dP) against dO . O")
O = np.array(rc["O"]); dO = np.array(j["dY"])
print(f"  P . dP = {(P[2,:3] * g['P'][2,:3]).sum():+.5f}; dO . O = {dO[2] @ O[2]:+.5f}; all rows: largest difference {np.abs((P[:, :5] * np.tril(g['P'])).sum(-1) - (dO * O).sum(-1)).max():.1e}")
print("E18.6  wiggling W_v[0, 1] with shrinking steps (central difference)")
m = hand_tiny(); W = m.blocks[0].att.W_v; old = W[0, 1]; true = g["W_v"][0, 1]
for h in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
    W[0, 1] = old + h; up = m.loss(IDS); W[0, 1] = old - h; dn = m.loss(IDS); W[0, 1] = old
    print(f"  h = {h:.0e}: {(up - dn)/(2*h):+.10f}, error {abs((up-dn)/(2*h) - true):.1e}")
print(f"  rule: {true:+.10f}")
print("E18.7  share tables kept at longer contexts (arithmetic, 32-bit)")
print(f"  12 x 4 x 2048 x 2048 x 4 bytes = {12*4*2048*2048*4/1e6:.0f} MB, four times the 201 MB at 1,024")
print(f"  12 x 4 x 8192 x 8192 x 4 bytes = {12*4*8192*8192*4/1e9:.1f} GB, {12*4*8192*8192*4/11.1e6:,.0f} times Tiny's 11.1 MB of learned numbers")
print("E18.8  the four-key example with the mean key taken out of the keys first")
eta = 3/1024; Kx = np.array([[65536.0, j - 1.0] for j in range(4)]); gx = np.array([1+eta, -1, -eta, 0])
gb = torch.tensor(gx, dtype=torch.float32).to(torch.bfloat16).double().numpy()
Kc = Kx - Kx.mean(0)
print(f"  rounded nudge times raw keys: {row(gb @ Kx / 8, 6)}; times centred keys: {row(gb @ Kc / 8, 6)}; exact {row(gx @ Kx / 8, 6)}")
print("E18.9  layer 1 of the hand Tiny: dW_o = O^T dY")
L, k = traced(m, IDS, layer=1); L.backward()
O = npy(k["O"]); dY = npy(k["Y"].grad)
G = torch_grads(m, IDS)[1]
print(f"  O^T dY = {np.round(O.T @ dY, 5).tolist()}; autograd {np.round(G['h.1.att.W_o'], 5).tolist()}")
print("E18.10 layer norm's backward against autograd (seed 18, a row of 8)")
rng = np.random.default_rng(18); x = rng.standard_normal(8); gam = rng.standard_normal(8); dz = rng.standard_normal(8)
mu = x.mean(); s = np.sqrt(((x - mu) ** 2).mean() + 1e-5); xh = (x - mu) / s; u = dz * gam
hand = (u - u.mean() - xh * (u * xh).mean()) / s
xt = torch.tensor(x, requires_grad=True)
zt = torch.nn.functional.layer_norm(xt, (8,), weight=torch.tensor(gam), eps=1e-5); zt.backward(torch.tensor(dz))
print(f"  by hand {row(hand)}; largest difference from autograd {np.abs(hand - xt.grad.numpy()).max():.1e}")
