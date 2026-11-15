"""Chapter 18, box "Where this bites" (low precision): the rows of dS add to zero, so dQ = c dS K ignores any
offset shared by every key.  Rounding dS to bfloat16 before the product breaks the zero sum, and the shared
offset leaks into dQ.  Part 1 rebuilds the four-key example of "Broken Symmetry in BF16 Attention" (arXiv
2609.34272): query 0, keys (65536, j - 1), scale 1/8, score nudge (1 + eta, -1, -eta, 0), eta = 3/1024.
Part 2: a seeded random head (64 keys, 64 wide) with a shared key offset of growing size; the error in dQ with dS
rounded to bfloat16, and with the rounded row's mean taken back out (32-bit) before the product.
Seed 18; PyTorch's bfloat16 rounding; products added in 32-bit."""
import _common
import numpy as np, torch
from _common import save_json
eta = 3 / 1024
K = np.array([[65536.0, j - 1.0] for j in range(4)])
g = np.array([1 + eta, -1.0, -eta, 0.0])
c = 1 / 8
exact = c * g @ K
gb = torch.tensor(g, dtype=torch.float32).to(torch.bfloat16).to(torch.float32).numpy().astype(np.float64)
print("Part 1: four keys")
print(f"  score nudge {g.tolist()}, adds to {g.sum():+.1e}")
print(f"  in bfloat16 it becomes {gb.tolist()}, which adds to {gb.sum():+.6f}")
print(f"  dQ exact: ({exact[0]:+.6f}, {exact[1]:+.6f})")
leak = c * gb @ K
print(f"  dQ from the rounded nudge: ({leak[0]:+.6f}, {leak[1]:+.6f})")
proj = gb - gb.mean()
pr = c * proj @ K
print(f"  rounded nudge with its mean taken back out: ({pr[0]:+.6f}, {pr[1]:+.6f})")
print("  the first coordinate's error is the row sum x the shared key part x the scale:"
      f" {gb.sum():+.6f} x 65536 x 1/8 = {gb.sum() * 65536 / 8:+.4f}")

print("\nPart 2: a random head, 64 keys, 64 wide, keys = shared offset + noise")
rng = np.random.default_rng(18)
n, d = 64, 64
q = rng.standard_normal((n, d)); k0 = rng.standard_normal((n, d)); v = rng.standard_normal((n, d)); dO = rng.standard_normal((n, d))
u = rng.standard_normal(d); u /= np.linalg.norm(u)
rows = []
print(f"  {'offset':>8} {'rel. error, rounded':>20} {'rel. error, mean removed':>25}")
for off in (0, 1, 4, 16, 64, 256, 1024):
    Kx = k0 + off * u
    S = q @ Kx.T / np.sqrt(d); S += np.triu(np.full((n, n), -np.inf), 1)
    P = np.exp(S - S.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    dP = dO @ v.T; dS = P * (dP - (P * dP).sum(-1, keepdims=True))
    dQ = dS @ Kx / np.sqrt(d)
    b = torch.tensor(dS, dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
    Kt = torch.tensor(Kx, dtype=torch.float32)
    dQb = (b @ Kt).numpy() / np.sqrt(d)
    mask = torch.tril(torch.ones(n, n))
    bm = b - (b.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)) * mask
    dQp = (bm @ Kt).numpy() / np.sqrt(d)
    e1 = float(np.linalg.norm(dQb - dQ) / np.linalg.norm(dQ)); e2 = float(np.linalg.norm(dQp - dQ) / np.linalg.norm(dQ))
    rows.append(dict(offset=off, rounded=e1, removed=e2))
    print(f"  {off:>8} {e1:>20.2e} {e2:>25.2e}")
save_json("bf16_leak", dict(four=dict(exact=exact.tolist(), leak=leak.tolist(), proj=pr.tolist(), gb=gb.tolist()), random=rows))
