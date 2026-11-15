"""Chapter 19, section 19.10: how much a norm multiplies the nudge passing back through it, for rows of
different sizes.  A row of 128 numbers with spread r goes through RMSNorm (and LayerNorm) with eps; a nudge of
spread 1 arrives on the output; the spread of the nudge leaving on the input is reported.  64-bit autograd.
The limit for a row far below sqrt(eps) is 1/sqrt(eps): 316 at eps 1e-5, 1,000 at eps 1e-6."""
import _common
import numpy as np, torch
from _common import SEED, save_json
torch.set_num_threads(1)
rng = np.random.default_rng(SEED)
z = rng.normal(0, 1, size=(64, 128)); z = z / np.sqrt((z ** 2).mean(-1, keepdims=True))
d = torch.tensor(rng.normal(0, 1, size=(64, 128)))
def rmsnorm(x, eps): return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps)
def layernorm(x, eps):
    m = x.mean(-1, keepdim=True); return (x - m) / torch.sqrt(((x - m) ** 2).mean(-1, keepdim=True) + eps)
out = {}
print("row spread r   RMSNorm eps 1e-5   eps 1e-6   |  LayerNorm eps 1e-5   (output spread at eps 1e-5)")
for r in (1.0, 0.1, 0.02, 0.01, 1e-3, 1e-4, 1e-6, 1e-12):
    row = []
    for f, eps in ((rmsnorm, 1e-5), (rmsnorm, 1e-6), (layernorm, 1e-5)):
        x = torch.tensor(z * r, requires_grad=True)
        y = f(x, eps); (y * d).sum().backward()
        row.append(float(x.grad.pow(2).mean().sqrt() / d.pow(2).mean().sqrt()))
    with torch.no_grad():
        ys = float(rmsnorm(torch.tensor(z * r), 1e-5).pow(2).mean().sqrt())
    out[f"{r:g}"] = row + [ys]
    print(f"  {r:9.0e}     {row[0]:12.4g}   {row[1]:9.4g}   |  {row[2]:12.4g}          {ys:.4g}")
print(f"  1/sqrt(1e-5) = {1/np.sqrt(1e-5):.1f};  1/sqrt(1e-6) = {1/np.sqrt(1e-6):.1f}")
print(f"  twelve layers each multiplying by 1,000: {1000.0 ** 12:.0e}; the 32-bit largest number {float(torch.finfo(torch.float32).max):.3g}")
save_json("norm_floor", out)
