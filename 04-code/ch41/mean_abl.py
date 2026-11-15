"""Chapter 41, 41.4: zeroing against mean replacement.  Same grown copy and test as induct.py (20 runs of 60 random tokens shown twice, seed 1865).
Each head's mean output before W_o is taken over every seat of 20 DIFFERENT runs of random tokens shown twice (seed 1865 + 11), then a head is
removed by writing that mean in place of its output instead of zero.  CPU, one thread, 32-bit."""
from lib import *
import numpy as np
torch.set_num_threads(1)
tt = load(CH24); sd = torch.load(os.path.join(HERE, "outputs", "ckpt", "repeat_1e-3_1500.pt"))
for k in tt.t: tt.t[k] = sd[k].float().detach()
N = 60; NL, H = 12, 4
def mk(seed):
    rng = np.random.default_rng(seed); return torch.tensor(np.stack([np.concatenate([r, r]) for r in (rng.integers(256, 2048, size=N) for _ in range(20))]))
X = mk(SEED); Xm = mk(SEED + 11)
grab = {}
with torch.no_grad(): forward(tt, Xm[:, :-1], grab=grab)
meanz = {(l, h): grab[l][:, h].mean(dim=(0, 1)) for l in range(NL) for h in range(H)}
def rep(**kw):
    with torch.no_grad(): lg = forward(tt, X[:, :-1], **kw)["logits"]
    L = torch.nn.functional.cross_entropy(lg.reshape(-1, 2048), X[:, 1:].reshape(-1), reduction="none").reshape(len(X), -1)
    return float(L[:, N - 1:].mean())
base = rep(); print(f"repeat loss, nothing removed: {base:.3f}")
sets = {"10.0": {(10, 0)}, "11.1": {(11, 1)}, "11.2": {(11, 2)}, "1.2": {(1, 2)}, "0.2": {(0, 2)}, "6.2": {(6, 2)},
        "10.0 + 11.1": {(10, 0), (11, 1)}, "10.0 + 11.1 + 11.2": {(10, 0), (11, 1), (11, 2)}, "10.0 + 11.1 + 11.2 + 11.0": {(10, 0), (11, 1), (11, 2), (11, 0)}}
print(f"{'removed':28s} {'zeroed':>8s} {'mean-replaced':>14s}")
out = {}
for nm, S in sets.items():
    z = rep(ablate=S); m = rep(ablate=S, meanz=meanz); out[nm] = (z, m); print(f"{nm:28s} {z:8.3f} {m:14.3f}")
rr = np.random.default_rng(SEED + 3); zs = []; ms = []
for _ in range(10):
    i = int(rr.integers(NL * H)); S = {(i // H, i % H)}; zs.append(rep(ablate=S)); ms.append(rep(ablate=S, meanz=meanz))
print(f"{'ten random single heads':28s} {np.mean(zs):8.3f} {np.mean(ms):14.3f}   (mean; ranges {min(zs):.3f} to {max(zs):.3f} and {min(ms):.3f} to {max(ms):.3f})")
save_json("mean_abl", dict(base=base, sets=out, rand_zero=zs, rand_mean=ms))
