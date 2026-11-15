"""Chapter 25 (copied from Chapter 24's train_tiny.py; only the WSD tag changed so that runs with weight decay
or another reading order do not overwrite Chapter 24's tables): the full Tiny (twelve layers, width 128, 2,766,592 learned numbers, Chapter 14's starts, seed 1865)
trained on the Carroll pool through ShardStream (Chapter 23's 24 shards, buffer 256), 8 windows of 129 a step
(1,024 predictions).  32-bit throughout (Chapter 23 found mixed-BF16 runs do not repeat; 32-bit ones do).  Adam,
b2 0.95, no weight decay, warmup 50 steps.  One thread.

  python3 train_tiny.py cosine LR STEPS          warmup then cosine to 0.1 of LR over STEPS (the sweep and the references)
  python3 train_tiny.py wsd LR TRUNK B1,B2,...   warmup then LR held constant for TRUNK steps; at each branch point B a
                                                 copy of the run (tables and Adam's averages) is decayed in a straight
                                                 line to zero over B/4 more steps, giving a finished run of 1.25 B steps
Logged: the batch loss and the global nudge norm at every step; every 50 steps the size of one step against the
size of the tables (rms of the change / rms of the tables, all tables together); held-out loss (Carroll held-out,
Looking-Glass) every 100 steps and at the end of every run.  Writes outputs/run_TAG.json; branch tables to
outputs/ckpt/ (not in the code zip)."""
import _common
import sys, time, datetime, json, os, copy
import numpy as np
import torch
torch.set_num_threads(1)
torch.manual_seed(1865)
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from tiny.optim import Adam
from tiny.train import global_norm
from tiny import data as D
from _common import SEED, HERE, sentence_ids
from _pool import heldout_sets, shards, WINDOWS_PER_PASS
import math
MODE, LR = sys.argv[1], float(sys.argv[2])
WU = 50
G, HB = (torch.tensor(a) for a in heldout_sets())
SENT = torch.tensor(sentence_ids())
os.makedirs(os.path.join(HERE, "outputs", "ckpt"), exist_ok=True)

def held(tt):
    with torch.no_grad():
        return float(tt.loss(HB)), float(tt.loss(G)), float(tt.loss(SENT))

FLOOR = float(os.environ.get("FLOOR", "0.1"))       # where the cosine ends, as a fraction of LR
def cosine(total):
    return lambda s: (s + 1) / WU if s < WU else FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * (s - WU) / max(1, total - WU)))

def run(tt, opt, stream, s0, s1, sched, log, evals, every_eval=100):
    """Steps s0 .. s1-1.  Appends (step, loss, norm) to log['steps'] and (step, carroll, glass, sentence) to evals."""
    for s in range(s0, s1):
        X = torch.tensor(stream.next_batch())
        for v in tt.t.values():
            v.grad = None
        L = tt.loss(X); L.backward()
        Lf = float(L.detach())
        gn = global_norm([v.grad for v in tt.t.values()])
        mult = sched(s)
        if s % 50 == 0:
            before = {k: v.detach().clone() for k, v in tt.t.items()}
        opt.step(tt.t, mult)
        if s % 50 == 0:
            num = sum(float(((tt.t[k].detach() - before[k]).double() ** 2).sum()) for k in before)
            den = sum(float((before[k].double() ** 2).sum()) for k in before)
            log["ratio"].append((s, math.sqrt(num / den)))
        log["steps"].append((s, Lf, gn, mult))
        if not math.isfinite(Lf):
            break
        if (s + 1) % every_eval == 0:
            evals.append((s + 1,) + held(tt))
    return tt

m = Tiny()
tt = TorchTiny.from_numpy(m, dtype=torch.float32)
WD = float(os.environ.get("WD", "0"))             # AdamW decay (Chapter 20); 0 = plain Adam
opt = Adam(LR, wd=WD)
OSEED = int(os.environ.get("ORDER", SEED))          # reading-order seed (Chapter 23); starts stay seed 1865
stream = D.ShardStream(shards(), window=129, batch=8, buffer=256, seed=OSEED)
t0 = time.perf_counter()
out = dict(mode=MODE, lr=LR, wd=WD, floor=FLOOR, order_seed=OSEED, warmup=WU, date=str(datetime.date.today()), threads=torch.get_num_threads(),
           N_total=sum(m.count().values()))
if MODE == "cosine":
    STEPS = int(sys.argv[3]); TAG = f"cos_{sys.argv[2]}_{STEPS}" + (f"_wd{os.environ['WD']}" if WD else "") + (f"_floor{os.environ['FLOOR']}" if "FLOOR" in os.environ else "") + (f"_o{OSEED}" if OSEED != SEED else "")
    log = dict(steps=[], ratio=[]); ev = [(0,) + held(tt)]
    run(tt, opt, stream, 0, STEPS, cosine(STEPS), log, ev)
    if ev[-1][0] != STEPS:
        ev.append((STEPS,) + held(tt))
    out.update(steps=STEPS, log=log, evals=ev, final=ev[-1], passes=STEPS * 8 / WINDOWS_PER_PASS)
    print(f"{TAG}: held-out Carroll {ev[-1][1]:.4f} Looking-Glass {ev[-1][2]:.4f} sentence {ev[-1][3]:.4f}")
else:
    TRUNK = int(sys.argv[3]); BR = [int(b) for b in sys.argv[4].split(",")]
    TAG = f"wsd_{sys.argv[2]}_{TRUNK}" + (f"_wd{os.environ['WD']}" if WD else "") + (f"_o{OSEED}" if OSEED != SEED else "")
    log = dict(steps=[], ratio=[]); ev = [(0,) + held(tt)]
    const = lambda s: (s + 1) / WU if s < WU else 1.0
    branches = {}
    s = 0
    for b in sorted(BR):
        run(tt, opt, stream, s, b, const, log, ev); s = b
        # the branch: a copy of the tables, of Adam's averages and of the reading position
        bt = TorchTiny({k: v.detach().clone().numpy() for k, v in tt.t.items()}, tt.config, dtype=torch.float32)
        bo = copy.deepcopy(opt); bs = copy.deepcopy(stream)
        n = b // 4
        dec = lambda q, b=b, n=n: (b + n - q) / n          # 1 at the branch point, falling to 1/n at the last step
        blog = dict(steps=[], ratio=[]); bev = [(b,) + ev[-1][1:]] if ev[-1][0] == b else [(b,) + held(bt)]
        run(bt, bo, bs, b, b + n, dec, blog, bev, every_eval=max(1, n // 5))
        if bev[-1][0] != b + n:
            bev.append((b + n,) + held(bt))
        branches[b] = dict(start=b, length=b + n, log=blog, evals=bev, final=bev[-1], passes=(b + n) * 8 / WINDOWS_PER_PASS)
        torch.save({k: v.detach() for k, v in bt.t.items()}, os.path.join(HERE, "outputs", "ckpt", f"{TAG}_b{b}.pt"))
        print(f"{TAG} branch at {b} decayed to {b + n}: held-out Carroll {bev[-1][1]:.4f} Looking-Glass {bev[-1][2]:.4f} "
              f"sentence {bev[-1][3]:.4f}  (trunk at {b}: Carroll {ev[-1][1]:.4f})", flush=True)
    out.update(trunk=TRUNK, log=log, evals=ev, branches=branches)
dt = time.perf_counter() - t0
out.update(seconds=dt)
json.dump(out, open(os.path.join(HERE, "outputs", f"run_{TAG}.json"), "w"))
print(f"{TAG}: {dt:.0f} s")
