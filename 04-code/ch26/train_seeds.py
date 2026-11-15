"""Chapter 26: what moves a run.  The full Tiny (twelve layers, width 128, 2,766,592 learned numbers) trained for 400 steps on
the Carroll pool (ShardStream, 8 windows of 129 a step), 32-bit, AdamW (decay 0.1, Tiny's default since Chapter 25), b2 0.95,
lr 1e-3, warmup 50 steps then cosine to 0.1 of lr.  Everything is fixed except what the environment variables change:
  INIT=s      the seed of the starting tables (Tiny(seed=s)); default 1865
  ORDER=s     the seed of the reading order (ShardStream(seed=s)); default 1865
  THREADS=n   PyTorch's thread count (default 1)
  NUDGE=1     add one float32 step (one ulp) to a single starting number, E[0, 0], before training
  LR=x        the peak learning rate (default 1e-3)
  REP=name    a label only, for repeats of the same configuration
Logs the batch loss every step and held-out loss (Carroll held-out, Looking-Glass) every 50 steps; at the end the SHA-256
fingerprint of all tables (tiny/seed.py).  Writes outputs/seeds_TAG.json and the final tables to outputs/ckpt/ (not zipped)."""
import os, sys
TH = int(os.environ.get("THREADS", "1"))
os.environ["OMP_NUM_THREADS"] = str(TH); os.environ["MKL_NUM_THREADS"] = str(TH); os.environ["OPENBLAS_NUM_THREADS"] = str(TH)
import _common
import time, json, math, datetime
import numpy as np
import torch
torch.set_num_threads(TH)
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from tiny.optim import Adam
from tiny import data as D
from tiny.seed import fingerprint, run_card
from _common import HERE
from _pool import heldout_sets, shards
INIT = int(os.environ.get("INIT", "1865")); ORDER = int(os.environ.get("ORDER", "1865"))
NUDGE = os.environ.get("NUDGE") == "1"; REP = os.environ.get("REP", "")
STEPS, WU, WD = 400, 50, 0.1
LR = float(os.environ.get("LR", "1e-3"))
TAG = f"i{INIT}_o{ORDER}_t{TH}" + ("_nudge" if NUDGE else "") + (f"_lr{os.environ['LR']}" if "LR" in os.environ else "") + (f"_{REP}" if REP else "")
G, HB = (torch.tensor(a) for a in heldout_sets())
def held(tt):
    with torch.no_grad():
        return float(tt.loss(HB)), float(tt.loss(G))
sched = lambda s: (s + 1) / WU if s < WU else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - WU) / (STEPS - WU)))
m = Tiny(seed=INIT)
tt = TorchTiny.from_numpy(m, dtype=torch.float32)
if NUDGE:
    with torch.no_grad():
        e = tt.t["E"]; e[0, 0] = torch.nextafter(e[0, 0], torch.tensor(float("inf")))
start_fp = fingerprint(tt.t)
opt = Adam(LR, wd=WD)
stream = D.ShardStream(shards(), window=129, batch=8, buffer=256, seed=ORDER)
losses, evals = [], [(0,) + held(tt)]
t0 = time.perf_counter()
for s in range(STEPS):
    X = torch.tensor(stream.next_batch())
    for v in tt.t.values():
        v.grad = None
    L = tt.loss(X); L.backward()
    losses.append(float(L.detach()))
    opt.step(tt.t, sched(s))
    if (s + 1) % 50 == 0:
        evals.append((s + 1,) + held(tt))
dt = time.perf_counter() - t0
out = dict(tag=TAG, init=INIT, order=ORDER, threads=TH, nudge=NUDGE, rep=REP, steps=STEPS, lr=LR, wd=WD, warmup=WU,
           date=str(datetime.date.today()), seconds=dt, start_fingerprint=start_fp, fingerprint=fingerprint(tt.t),
           losses=losses, evals=evals, final=evals[-1], card=run_card())
json.dump(out, open(os.path.join(HERE, "outputs", f"seeds_{TAG}.json"), "w"))
torch.save({k: v.detach() for k, v in tt.t.items()}, os.path.join(HERE, "outputs", "ckpt", f"seeds_{TAG}.pt"))   # not in the code zip
print(f"{TAG}: Carroll {evals[-1][1]:.4f} Looking-Glass {evals[-1][2]:.4f} fp {out['fingerprint']} {dt:.0f} s", flush=True)
