"""Chapter 19, section 19.8: gradient clipping by global norm.  Part 1 repeats Chapter 17's run that went to nan
(Tiny's default start, plain SGD, learning rate 1.0, the same eight windows per step) with and without clipping at
1.0, for 400 steps.  Part 2 shows the 32-bit overflow inside a global-norm clip: one nudge above about 1.84e19
squares to infinity, the norm becomes inf, and the clip multiplies every nudge by zero without an error."""
import _common
import time, datetime
import numpy as np
import torch
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from tiny.train import train_torch
from _common import novel_ids, heldout_ids, tok, save_json, SEED
torch.set_num_threads(8)
T = tok()
nov = np.array(novel_ids(T)); held = np.array(heldout_ids(T))
STEPS = 400
rng = np.random.default_rng(SEED)
starts = rng.integers(0, len(nov) - 129, size=(STEPS, 8))
hstarts = rng.integers(0, len(held) - 129, size=16)
H = torch.tensor(np.stack([held[a:a + 129] for a in hstarts]))
def batches(s):
    return torch.tensor(np.stack([nov[a:a + 129] for a in starts[s]]))
print(f"{datetime.date.today()}; seed {SEED}; threads {torch.get_num_threads()}")
print("Part 1: learning rate 1.0 from Tiny's default start, 400 steps")
out = {}
for clip in (None, 1.0, 5.0):
    tt = TorchTiny.from_numpy(Tiny(), dtype=torch.float32)
    norms = []
    t0 = time.perf_counter()
    hist = train_torch(tt, batches, STEPS, 1.0, clip=clip, norms=norms)
    dt = time.perf_counter() - t0
    with torch.no_grad():
        hend = float(tt.loss(H)) if len(hist) == STEPS else float("nan")
    clipped = sum(1 for n in norms if clip is not None and n > clip)
    out[str(clip)] = dict(train=[L for _, L in hist], norms=norms, stopped=len(hist), held_end=hend, clipped=clipped)
    name = "no clipping" if clip is None else f"clip at {clip:g}"
    if len(hist) < STEPS:
        tail = ", ".join(f"{L:.4g}" for _, L in hist[-4:])
        print(f"  {name:12s} stopped at step {len(hist) - 1}; last batch losses {tail}; nudge norms before that "
              + ", ".join(f"{n:.3g}" for n in norms[-4:]))
    else:
        print(f"  {name:12s} 400 steps; batch loss {hist[0][1]:.4f} -> {np.mean([L for _, L in hist[-25:]]):.4f} (last 25);"
              f" held-out {hend:.4f}; clipped on {clipped} of 400 steps; nudge norm median {np.median(norms):.3g},"
              f" largest {max(norms):.3g}; {dt:.0f} s")
print("\nPart 2: the clip that zeroes everything (32-bit)")
for big in (1e18, 1e19, 2e19, 1e20):
    a = torch.zeros(1000, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(3, dtype=torch.float32, requires_grad=True)
    a.grad = torch.full((1000,), 0.01, dtype=torch.float32)
    b.grad = torch.tensor([big, 0.5, -0.5], dtype=torch.float32)
    total = torch.nn.utils.clip_grad_norm_([a, b], max_norm=1.0)
    sq = torch.tensor(big, dtype=torch.float32) ** 2
    print(f"  one nudge of {big:.0e}: its square in 32-bit {float(sq):.4g}; norm returned {float(total):.4g};"
          f" a's nudges after the clip {float(a.grad[0]):.4g} (were 0.01); b's other two {float(b.grad[1]):.4g}, {float(b.grad[2]):.4g}")
    out[f"overflow_{big:.0e}"] = dict(norm=float(total), a_after=float(a.grad[0]), b_after=[float(x) for x in b.grad])
gn64 = float(torch.sqrt((torch.full((1000,), 0.01, dtype=torch.float64) ** 2).sum() + torch.tensor([2e19, 0.5, -0.5], dtype=torch.float64).pow(2).sum()))
print(f"  the same norm in 64-bit for the 2e19 case: {gn64:.4g}; scale factor 1/norm {1/gn64:.3g}, so a's nudges would become {0.01/gn64:.3g}")
out["fp64_2e19"] = gn64
print(f"  largest 32-bit number {torch.finfo(torch.float32).max:.4g}; its square root {np.sqrt(float(torch.finfo(torch.float32).max)):.4g}")
save_json("clipping", out)
