"""Chapter 21, Exercise 11: Tiny trained for 400 steps with every product's inputs rounded to FP8 E4M3,
with one scale per table ("tensor") or with MX blocks of 32 along the summed axis ("mx").  Same windows, Adam 1e-3,
warmup 40 + cosine to 0.1, one thread.  An emulation: the products are computed in 32-bit from FP8-rounded inputs.
Rounding is applied in the forward pass only; the backward pass sees the rounding as the identity (a
straight-through estimate), which is how emulated low-precision training is usually written.
Usage: python3 ex11_fp8.py tensor|mx   (writes outputs/ex11_fp8_MODE.json)"""
import _common
import sys, json, os, time
import numpy as np
import torch
torch.set_num_threads(1)
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from tiny.optim import Adam, warmup_cosine
from tiny.precision import MixedTiny, train_mixed, scaled_round, mx_round
from _common import novel_ids, heldout_ids, tok, SEED, HERE
MODE = sys.argv[1]
def q(x, axis):
    with torch.no_grad():
        if MODE == "tensor":
            r = scaled_round(x.detach(), "FP8 E4M3")
        else:
            xt = x.detach().transpose(axis, -1).contiguous()
            r = mx_round(xt, 32, "FP8 E4M3").transpose(axis, -1)
    return x + (r - x).detach()                     # forward: rounded value; backward: identity
class FP8Tiny(MixedTiny):
    def mm(self, a, b):
        return q(a, -1) @ q(b, -2)
T = tok(); nov = np.array(novel_ids(T)); held = np.array(heldout_ids(T))
rng = np.random.default_rng(SEED)
starts = rng.integers(0, len(nov) - 129, size=(400, 8)); hs = rng.integers(0, len(held) - 129, size=16)
H = torch.tensor(np.stack([held[a:a + 129] for a in hs]))
batches = lambda s: torch.tensor(np.stack([nov[a:a + 129] for a in starts[s]]))
base = MixedTiny.from_numpy(Tiny(), compute=torch.float32)
tt = FP8Tiny.__new__(FP8Tiny); tt.config, tt.dtype, tt.t, tt.compute = base.config, torch.float32, base.t, torch.float32
ref = lambda: float(TorchTiny({k: v.detach().numpy() for k, v in tt.t.items()}, tt.config, dtype=torch.float32).loss(H))
t0 = time.perf_counter()
h = train_mixed(tt, batches, 400, Adam(1e-3), warmup_cosine(40, 400, 0.1), mode="mixed")
with torch.no_grad():
    end = ref()
out = dict(mode=MODE, held_end=end, train_last25=float(np.mean([L for _, L in h[-25:]])), seconds=time.perf_counter() - t0)
json.dump(out, open(os.path.join(HERE, "outputs", f"ex11_fp8_{MODE}.json"), "w"), indent=1)
print(f"FP8 E4M3 products ({MODE} scales), 400 steps: held-out {end:.4f} (32-bit 5.2684); batch last 25 {out['train_last25']:.4f}; {out['seconds']:.0f} s")
