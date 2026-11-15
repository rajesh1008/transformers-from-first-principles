"""Chapter 17, section 17.9: plain gradient descent on the real Tiny (2,766,592 learned numbers), on the novel.
Each step: eight windows of 129 tokens drawn with seed 1865 (1,024 predictions), autograd nudges, one SGD step.
32-bit.  Eight threads for this script only (torch.set_num_threads(8)); the timings are wall clock, dated.
Held-out loss is measured on 16 fixed windows of the Looking-Glass text the model never trains on."""
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
STEPS, EVERY = 600, 25
rng = np.random.default_rng(SEED)
starts = rng.integers(0, len(nov) - 129, size=(STEPS, 8))
hstarts = rng.integers(0, len(held) - 129, size=16)
nstarts = rng.integers(0, len(nov) - 129, size=16)
H = torch.tensor(np.stack([held[a:a + 129] for a in hstarts]))
NV = torch.tensor(np.stack([nov[a:a + 129] for a in nstarts]))
def batches(s):
    return torch.tensor(np.stack([nov[a:a + 129] for a in starts[s]]))
def evals(tt):
    with torch.no_grad():
        return float(tt.loss(NV)), float(tt.loss(H))
print(f"{datetime.date.today()}; seed {SEED}; threads {torch.get_num_threads()}; novel {len(nov):,} tokens, held-out {len(held):,}")
print(f"{STEPS} steps x 8 windows x 128 predictions = {STEPS*1024:,} predictions = {STEPS*1024/len(nov):.1f} passes through the novel")
out = {}
for lr in (0.03, 0.1, 0.3, 0.6, 1.0):
    tt = TorchTiny.from_numpy(Tiny(), dtype=torch.float32)
    curve = []
    def on_step(s, L):
        if s % EVERY == 0 or s == STEPS - 1:
            curve.append((s, L) + evals(tt))
    t0 = time.perf_counter()
    hist = train_torch(tt, batches, STEPS, lr, every=1, on_step=on_step)
    dt = time.perf_counter() - t0
    blew = len(hist) < STEPS
    end = evals(tt)
    out[str(lr)] = dict(curve=curve, train=[L for _, L in hist], stopped=len(hist), seconds=dt, end=end)
    if blew:
        print(f"lr {lr}: stopped at step {len(hist)} with a loss that is not a finite number; last finite batch losses "
              + ", ".join(f"{L:.4g}" for _, L in hist[-4:]))
    else:
        print(f"lr {lr}: batch loss {hist[0][1]:.4f} -> {np.mean([L for _, L in hist[-25:]]):.4f} (average of last 25);"
              f" novel windows {curve[0][2]:.4f} -> {end[0]:.4f}; held-out {curve[0][3]:.4f} -> {end[1]:.4f}; {dt:.0f} s,"
              f" {1000*dt/STEPS:.0f} ms a step")
    if lr == 0.3:
        best = tt
print("\nafter 600 steps at lr 0.3, the running sentence, Tiny's top three next pieces at each seat:")
ids = T.encode("the cat sat on the mat")
with torch.no_grad():
    S = best.scores(torch.tensor(ids))
P = torch.softmax(S, -1).numpy()
for s in range(len(ids)):
    top = np.argsort(-P[s])[:3]
    tru = f" | true next {T.decode([ids[s+1]])!r} share {P[s, ids[s+1]]:.4f}" if s + 1 < len(ids) else ""
    print(f"  after {T.decode(ids[:s+1])!r}: " + ", ".join(f"{T.decode([int(i)])!r} {P[s,i]:.3f}" for i in top) + tru)
with torch.no_grad():
    print(f"running sentence loss: before {Tiny().loss(ids):.4f}, after {float(best.loss(torch.tensor(ids))):.4f}")
save_json("train_tiny", out)
