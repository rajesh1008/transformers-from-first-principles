"""Chapter 34, 34.6: prune Tiny, then let it recover.  Chapter 24's checkpoint (twelve layers, 1.9812 bits per character on
held-out Carroll) is pruned one of three ways, then trained for STEPS steps with the pruned positions held at zero (the mask is
re-applied after every step), on Chapter 24's data stream (8 windows of 129 a step, seed 1865), Adam, peak 3e-4, 20 warmup
steps then cosine to 0.1, 32-bit, one thread.
  python3 prune_distill.py PRUNE MODE STEPS
    PRUNE  none                (control: the unpruned checkpoint trained the same way)
           units50 | units75   whole feed-forward units removed in every layer (ranked by |W_1 col| x |W_2 row|)
           w80                 80% of each linear map's weights zeroed, smallest |w| within each output channel
    MODE   none (score after pruning only) | hard (cross-entropy) | kd1 (KL to the unpruned teacher's table, T = 1)
The teacher in kd1 is the same checkpoint, unpruned."""
import _common
import sys, time, math, json, os
import torch
torch.set_num_threads(1); torch.manual_seed(1865)
from tiny.optim import Adam
from tiny.eval import load
from tiny import data as D
from tiny.tokenizer import load_bpe
from _common import SEED, HERE, TINY_BPE
from _pool import shards, corpora
PRUNE, MODE, STEPS = sys.argv[1], sys.argv[2], int(sys.argv[3])
CK = os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")
teacher = load(CK); st = load(CK)
for v in st.t.values(): v.requires_grad_(True)
masks = {}
with torch.no_grad():
    if PRUNE.startswith("units"):
        frac = int(PRUNE[5:]) / 100
        for l in range(st.config["n_layers"]):
            p = f"h.{l}.ff."; W1, W2 = st.t[p + "W_1"], st.t[p + "W_2"]
            drop = (W1.norm(dim=0) * W2.norm(dim=1)).argsort()[:int(W1.shape[1] * frac)]
            m1 = torch.ones_like(W1); m1[:, drop] = 0; mb = torch.ones_like(st.t[p + "b_1"]); mb[drop] = 0
            m2 = torch.ones_like(W2); m2[drop, :] = 0
            masks[p + "W_1"], masks[p + "b_1"], masks[p + "W_2"] = m1, mb, m2
    elif PRUNE == "w80":
        for k, w in st.t.items():
            if w.dim() == 2 and k not in ("E", "P"):
                idx = w.abs().argsort(0)[:int(w.shape[0] * 0.8)]; m = torch.ones_like(w); m.scatter_(0, idx, 0.0); masks[k] = m
    for k, m in masks.items(): st.t[k].mul_(m)
total = sum(v.numel() for v in st.t.values())
kept = total - sum(int((m == 0).sum()) for m in masks.values())
tk = load_bpe(TINY_BPE); ids = [int(i) for i in corpora()["held_b"]]
def bpc(model):
    tot, chars = 0.0, 0
    for a in range(0, len(ids) - 1, 129):
        w = ids[a:a + 129]
        if len(w) < 2: break
        with torch.no_grad(): lp = torch.log_softmax(model.scores(w[:-1]).double(), -1)
        tot += float(-lp[torch.arange(len(w) - 1), torch.tensor(w[1:])].sum()); chars += len(tk.decode(w[1:]))
    return tot / math.log(2) / chars
b0 = bpc(st)
print(f"{PRUNE} {MODE}: nonzero numbers {kept:,} of {total:,}; teacher 1.9812 measured {bpc(teacher):.4f}; after pruning {b0:.4f}", flush=True)
curve = [(0, b0)]; t0 = time.perf_counter()
if MODE != "none":
    opt = Adam(3e-4, wd=0.0)
    stream = D.ShardStream(shards(), window=129, batch=8, buffer=256, seed=SEED)
    WU = 20
    sched = lambda s: (s + 1) / WU if s < WU else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - WU) / max(1, STEPS - WU)))
    for s in range(STEPS):
        X = torch.tensor(stream.next_batch())
        for v in st.t.values(): v.grad = None
        if MODE == "hard":
            L = st.loss(X)
        else:
            inp = X[:, :-1]
            with torch.no_grad(): pt = teacher.scores(inp)
            ls = torch.log_softmax(st.scores(inp), -1); lt = torch.log_softmax(pt, -1)
            L = (lt.exp() * (lt - ls)).sum(-1).mean()
        L.backward(); opt.step(st.t, sched(s))
        with torch.no_grad():
            for k, m in masks.items(): st.t[k].mul_(m)
        if (s + 1) % 100 == 0 or s + 1 == STEPS:
            b = bpc(st); curve.append((s + 1, b))
            print(f"  step {s + 1}: held-out Carroll {b:.4f} bits per character ({time.perf_counter() - t0:.0f} s)", flush=True)
zeros_ok = all(float((st.t[k] * (1 - m)).abs().max()) == 0.0 for k, m in masks.items())
out = dict(prune=PRUNE, mode=MODE, steps=STEPS, nonzero=kept, total=total, after_prune=b0, curve=curve, final=curve[-1][1],
           seconds=time.perf_counter() - t0, pruned_positions_still_zero=zeros_ok)
json.dump(out, open(os.path.join(HERE, "outputs", f"prune_distill_{PRUNE}_{MODE}_{STEPS}.json"), "w"))
print(f"{PRUNE} {MODE}: final {curve[-1][1]:.4f}; pruned positions still zero: {zeros_ok}; {out['seconds']:.0f} s")
