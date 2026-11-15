"""Chapter 19's exercise 11, done here: Chapter 19's start experiment (train_inits.py: plain gradient descent on Tiny,
400 steps of 8 windows of the novel, held-out on the 16 Looking-Glass windows, 32-bit) rerun at ONE thread for
three starts at learning rate 0.1, with the training windows drawn from seed 1865 (Chapter 19's) and from a second
seed, 7.  The held-out windows are Chapter 19's for both.  Usage: python3 inits_seed2.py SCHEME WINDOW_SEED"""
import _common
import sys, time, datetime, json, os
import numpy as np
import torch
torch.set_num_threads(1)
from tiny.model import Tiny
from tiny.init import initialise
from tiny.torch_tiny import TorchTiny
from tiny.train import train_torch
from _common import novel_ids, heldout_ids, tok, SEED, HERE
SCHEME, WSEED = sys.argv[1], int(sys.argv[2])
T = tok()
nov = np.array(novel_ids(T)); held = np.array(heldout_ids(T))
STEPS, LR = 400, 0.1
rng = np.random.default_rng(SEED)
starts19 = rng.integers(0, len(nov) - 129, size=(STEPS, 8))
hstarts = rng.integers(0, len(held) - 129, size=16)                # Chapter 19's held-out windows, unchanged
starts = starts19 if WSEED == SEED else np.random.default_rng(WSEED).integers(0, len(nov) - 129, size=(STEPS, 8))
H = torch.tensor(np.stack([held[a:a + 129] for a in hstarts]))
batches = lambda s: torch.tensor(np.stack([nov[a:a + 129] for a in starts[s]]))
m = initialise(Tiny(), SCHEME, 1.0)
tt = TorchTiny.from_numpy(m, dtype=torch.float32)
norms = []
t0 = time.perf_counter()
hist = train_torch(tt, batches, STEPS, LR, every=1, norms=norms)
with torch.no_grad():
    hend = float(tt.loss(H))
out = dict(scheme=SCHEME, window_seed=WSEED, lr=LR, steps=STEPS, held_end=hend, train=[L for _, L in hist],
           norms=norms, seconds=time.perf_counter() - t0, threads=1, date=str(datetime.date.today()))
json.dump(out, open(os.path.join(HERE, "outputs", f"inits_{SCHEME}_{WSEED}.json"), "w"))
print(f"{SCHEME:7s} window seed {WSEED:5d}: held-out {hend:.4f}; batch loss last 25 {np.mean(out['train'][-25:]):.4f}")
