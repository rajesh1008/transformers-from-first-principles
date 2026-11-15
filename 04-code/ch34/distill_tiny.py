"""Chapter 34, 34.2 and 34.3: distilling Tiny into a shallower Tiny.  Teacher: Chapter 24's checkpoint (twelve layers, width 128,
2,766,592 numbers, held-out Carroll 1.9812 bits per character).  Student: the same design with 4 layers (same width, same token
table size), Chapter 14's starts at seed 1865, trained for STEPS steps on Chapter 24's data stream (Carroll pool, 8 windows of 129
a step, the same reading order for every run), Adam, peak 1e-3, 50 warmup steps then cosine to 0.1, 32-bit, one thread.
  python3 distill_tiny.py MODE STEPS      MODE = hard | kd1 | kd2 | mix
    hard  loss = cross-entropy against the next token (the ordinary loss)
    kd1   loss = KL(teacher table || student table) at temperature 1
    kd2   loss = T^2 x KL at temperature 2 (both tables softened by T)
    mix   loss = 0.5 x cross-entropy + 0.5 x kd1
Reports held-out Carroll bits per character in windows of 129 (as Chapters 29 and 33), the Looking-Glass loss, and seconds."""
import _common
import sys, time, math, json, os
import numpy as np
import torch
torch.set_num_threads(1); torch.manual_seed(1865)
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from tiny.optim import Adam
from tiny.eval import load
from tiny import data as D
from _common import SEED, HERE, TINY_BPE
from _pool import shards, corpora
MODE, STEPS = sys.argv[1], int(sys.argv[2])
RSEED = int(os.environ.get("SEED", str(SEED)))   # tick 26: optional second seed (student start and reading order)
teacher = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
LAYERS = int(os.environ.get("LAYERS", "4"))
m = Tiny(n_layers=LAYERS, seed=RSEED)
st = TorchTiny.from_numpy(m, dtype=torch.float32)
nparams = sum(v.numel() for v in st.t.values())
opt = Adam(1e-3, wd=0.0)
stream = D.ShardStream(shards(), window=129, batch=8, buffer=256, seed=RSEED)
WU = 50
sched = lambda s: (s + 1) / WU if s < WU else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - WU) / max(1, STEPS - WU)))
def kl(ps, pt, T):
    ls = torch.log_softmax(ps / T, -1); lt = torch.log_softmax(pt / T, -1)
    return (lt.exp() * (lt - ls)).sum(-1).mean() * T * T
C = corpora()
from tiny.tokenizer import load_bpe
tk = load_bpe(TINY_BPE); ids = [int(i) for i in C["held_b"]]
def bpc(model):
    tot, chars = 0.0, 0
    for a in range(0, len(ids) - 1, 129):
        w = ids[a:a + 129]
        if len(w) < 2: break
        with torch.no_grad(): lp = torch.log_softmax(model.scores(w[:-1]).double(), -1)
        tot += float(-lp[torch.arange(len(w) - 1), torch.tensor(w[1:])].sum()); chars += len(tk.decode(w[1:]))
    return tot / math.log(2) / chars
print(f"{MODE}: student {LAYERS} layers, {nparams:,} numbers; teacher bpc {bpc(teacher):.4f}", flush=True)
t0 = time.perf_counter(); curve = []
for s in range(STEPS):
    X = torch.tensor(stream.next_batch())
    for v in st.t.values(): v.grad = None
    inp = X[:, :-1]
    if MODE == "hard":
        L = st.loss(X)
    else:
        with torch.no_grad(): pt = teacher.scores(inp)
        ps = st.scores(inp)
        if MODE == "kd1": L = kl(ps, pt, 1.0)
        elif MODE == "kd2": L = kl(ps, pt, 2.0)
        else: L = 0.5 * st.loss(X) + 0.5 * kl(ps, pt, 1.0)
    L.backward(); opt.step(st.t, sched(s))
    if (s + 1) % 250 == 0 or s + 1 == STEPS:
        b = bpc(st); curve.append((s + 1, b)); print(f"  step {s + 1}: held-out Carroll {b:.4f} bits per character ({time.perf_counter() - t0:.0f} s)", flush=True)
dt = time.perf_counter() - t0
out = dict(seed=RSEED, mode=MODE, steps=STEPS, layers=LAYERS, numbers=nparams, teacher=bpc(teacher), curve=curve, final=curve[-1][1], seconds=dt)
json.dump(out, open(os.path.join(HERE, "outputs", f"distill_{MODE}_{LAYERS}L_{STEPS}" + ("" if RSEED == SEED else f"_s{RSEED}") + ".json"), "w"))
print(f"{MODE}: final {curve[-1][1]:.4f}; {dt:.0f} s")
