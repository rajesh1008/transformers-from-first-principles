"""Chapter 22: Tiny trained on different data at a fixed budget of 384 steps x 8 windows of 129 tokens
(3,072 windows, 393,216 predictions; the Carroll pool holds 3,074 windows, so 384 steps is the longest run that reads
it once).  Mixed BF16 with a 32-bit master copy (Chapter 21's default), Adam lr 1e-3, warmup 40 and cosine to 0.1
(or constant 1e-3 for the "const_" runs).  Held-out loss in 32-bit every 32 steps on two sets: Looking-Glass
(Chapter 19's 16 windows) and the Carroll held-out (16 windows of the 4,000 held-out tokens).  One thread per run.
Usage: python3 train_data.py RUN [ORDER_SEED]   (writes outputs/train_data_RUN.json, or _RUN_sSEED.json)
  rep{P}      a fixed random slice of 3,072 / P pool windows, read P times in fresh orders (P = 1, 2, 4, 8, 32)
  mix{W}      each window drawn from Alice with probability W/100, else from the pool, at a random start
  cur_easy, cur_hard, cur_inorder   the rep1 windows ordered by unigram surprise (easy first, hard first) or in
              book order (no shuffling); const_random, const_easy: rep1 and cur_easy at a constant lr 1e-3"""
import _common
import sys, time, datetime, json, os
import numpy as np
import torch
torch.set_num_threads(1)
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from tiny.optim import Adam, warmup_cosine, constant
from tiny.precision import MixedTiny, train_mixed
from tiny import data as D
from _common import SEED, HERE
from _pool import corpora, pool_windows, heldout_sets
RUN = sys.argv[1]
OSEED = int(sys.argv[2]) if len(sys.argv) > 2 else SEED      # the seed of the reading order (and mixture draw) only
TAG = RUN + (f"_s{OSEED}" if OSEED != SEED else "")
STEPS, B, EVERY, WU, LR = 384, 8, 32, 40, 1e-3
c = corpora()
W, _ = pool_windows()
G, HB = heldout_sets()
G, HB = torch.tensor(G), torch.tensor(HB)
rng = np.random.default_rng(SEED + 22)
perm = rng.permutation(len(W))
def surprise_order(win):
    counts = np.bincount(c["pool"], minlength=2048).astype(float)
    return np.argsort(D.unigram_surprise(win, counts), kind="stable")
read = None
if RUN.startswith("rep"):
    P = int(RUN[3:])
    sl = W[np.sort(perm[:STEPS * B // P])]
    X = D.fixed_passes(sl, P, STEPS, B, seed=OSEED)
    read = sl
elif RUN.startswith("mix"):
    w = int(RUN[3:]) / 100
    streams = [c["alice"], c["pool"]]
    st = D.mixture_starts([len(s) for s in streams], [w, 1 - w], STEPS * B, seed=OSEED)
    X = np.stack([streams[k][a:a + 129] for k, a in st]).reshape(STEPS, B, 129)
    read = None
else:
    sl = W[np.sort(perm[:STEPS * B])]                      # the rep1 windows, in book order
    if RUN in ("cur_easy", "const_easy"):
        X = sl[surprise_order(sl)]
    elif RUN == "cur_hard":
        X = sl[surprise_order(sl)[::-1]]
    elif RUN == "cur_inorder":
        X = sl
    elif RUN == "const_random":
        X = D.fixed_passes(sl, 1, STEPS, B, seed=OSEED).reshape(-1, 129)
    X = X.reshape(STEPS, B, 129)
    if RUN in ("cur_easy", "cur_hard", "const_easy"):     # shuffle inside each batch only (order of steps kept)
        pass
    read = sl
X = torch.tensor(X)
m = Tiny()
tt = MixedTiny.from_numpy(m, compute=torch.bfloat16)
opt = Adam(LR)
def held(H):
    ref = TorchTiny({k: v.detach().to(torch.float32).numpy() for k, v in tt.t.items()}, tt.config, dtype=torch.float32)
    with torch.no_grad():
        return float(ref.loss(H))
R = torch.tensor(read[np.random.default_rng(SEED + 3).choice(len(read), 16, replace=False)]) if read is not None else None
curve = []
def on_step(s, L):
    if s % EVERY == 0 or s == STEPS - 1:
        curve.append((s, L, held(G), held(HB), held(R) if R is not None else None))
h0 = (held(G), held(HB))
t0 = time.perf_counter()
sched = constant() if RUN.startswith("const") else warmup_cosine(WU, STEPS, 0.1)
hist = train_mixed(tt, lambda s: X[s], STEPS, opt, sched, mode="mixed", on_step=on_step)
dt = time.perf_counter() - t0
out = dict(run=RUN, date=str(datetime.date.today()), seed=SEED, threads=torch.get_num_threads(), steps=STEPS,
           lr=LR, schedule="constant" if RUN.startswith("const") else "warmup 40 + cosine to 0.1",
           held0=h0, glass=curve[-1][2], carroll=curve[-1][3], read_loss=curve[-1][4],
           best_glass=min(x[2] for x in curve), best_carroll=min(x[3] for x in curve),
           curve=curve, train=[L for _, L in hist], seconds=dt,
           distinct_windows=int(len(read)) if read is not None else None)
json.dump(out, open(os.path.join(HERE, "outputs", f"train_data_{TAG}.json"), "w"), indent=1)
last = np.mean(out["train"][-24:])
rl = f"; on windows it read {out['read_loss']:.4f}" if out["read_loss"] is not None else ""
print(f"{TAG:13s} held-out Looking-Glass {out['glass']:.4f} (lowest {out['best_glass']:.4f}), Carroll {out['carroll']:.4f} "
      f"(lowest {out['best_carroll']:.4f}); batch last 24 {last:.4f}{rl}; {dt:.0f} s")
