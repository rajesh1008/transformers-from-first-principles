"""Chapter 19, section 19.7: does the starting size matter once training starts?  Plain gradient descent on the
real Tiny (2,766,592 learned numbers) from five starts, at two learning rates, 400 steps each.
Each step: eight windows of 129 tokens of the novel drawn with seed 1865 (the same windows for every run),
autograd nudges, one SGD step.  32-bit.  Eight threads for this script only.  Held-out loss on 16 fixed windows
of the Looking-Glass text.  The global norm of the nudges is recorded at every step (64-bit sum)."""
import _common
import time, datetime
import numpy as np
import torch
from tiny.model import Tiny
from tiny.init import initialise
from tiny.torch_tiny import TorchTiny
from tiny.train import train_torch
from _common import novel_ids, heldout_ids, tok, save_json, SEED
torch.set_num_threads(8)
T = tok()
nov = np.array(novel_ids(T)); held = np.array(heldout_ids(T))
STEPS, EVERY = 400, 25
rng = np.random.default_rng(SEED)
starts = rng.integers(0, len(nov) - 129, size=(STEPS, 8))
hstarts = rng.integers(0, len(held) - 129, size=16)
H = torch.tensor(np.stack([held[a:a + 129] for a in hstarts]))
def batches(s):
    return torch.tensor(np.stack([nov[a:a + 129] for a in starts[s]]))
STARTS = [("gpt2", 1.0, "0.02 everywhere (Tiny's default)"), ("scaled", 1.0, "0.02, W_o and W_2 / sqrt(24)"),
          ("fan_in", 1.0, "1/sqrt(fan-in)"), ("gpt2", 10.0, "0.2, ten times too large"), ("zeros", 1.0, "layer tables all zero")]
print(f"{datetime.date.today()}; seed {SEED}; threads {torch.get_num_threads()}; {STEPS} steps x 8 windows x 128 predictions")
out = {}
for lr in (0.1, 0.6):
    for scheme, scale, desc in STARTS:
        m = initialise(Tiny(), scheme, scale)
        tt = TorchTiny.from_numpy(m, dtype=torch.float32)
        curve, norms = [], []
        def on_step(s, L):
            if s % EVERY == 0 or s == STEPS - 1:
                with torch.no_grad():
                    curve.append((s, L, float(tt.loss(H))))
        with torch.no_grad():
            h0 = float(tt.loss(H))
        t0 = time.perf_counter()
        hist = train_torch(tt, batches, STEPS, lr, every=1, on_step=on_step, norms=norms)
        dt = time.perf_counter() - t0
        key = f"{scheme}x{scale:g}@{lr}"
        with torch.no_grad():
            hend = float(tt.loss(H)) if len(hist) == STEPS else float("nan")
        out[key] = dict(desc=desc, lr=lr, curve=curve, train=[L for _, L in hist], norms=norms, stopped=len(hist),
                        held0=h0, held_end=hend, seconds=dt)
        last = np.mean([L for _, L in hist[-25:]])
        if len(hist) < STEPS:
            print(f"lr {lr}  {desc:34s} stopped at step {len(hist)}: loss not finite; nudge norm at step 0 {norms[0]:.3g}")
        else:
            print(f"lr {lr}  {desc:34s} held-out {h0:.4f} -> {hend:.4f}; batch loss last 25 {last:.4f};"
                  f" nudge norm step 0 {norms[0]:.3g}, largest {max(norms):.3g}; {dt:.0f} s")
save_json("train_inits", out)
