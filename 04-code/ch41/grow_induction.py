"""[Chapter 41 copy of ../ch36/continue_repeat.py with a checkpoint save added at the end.]
Chapter 36, 36.3: can more training give Tiny an induction head?  Continue Chapter 24's Tiny for 600 AdamW steps
(lr 3e-4, 8 windows of 129 tokens per step, seed 1865) in one of two ways:
  repeat  each window is a 64-token passage of the Carroll pool followed by the same passage again (129 tokens in all)
  plain   ordinary 129-token windows of the pool (the control: the same amount of extra training, no planted repeats)
Every 100 steps: the best induction score and best previous-token score over Tiny's 48 heads (5 draws of 60 random
tokens shown twice), Tiny's loss on the first and second showing of 10 Looking-Glass passages of 60 tokens, and its loss
on 20 ordinary 129-token Looking-Glass windows.  CPU, one thread, 32-bit.   python3 continue_repeat.py MODE"""
from _common import *
import sys, math, time, torch, numpy as np
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
from _pool import corpora
MODE = sys.argv[1]; STEPS = int(os.environ.get("STEPS", "600"))
LR = float(os.environ.get("LR", "3e-4")); PLEN = int(os.environ.get("PLEN", "64")); EVERY = int(os.environ.get("EVERY", "100"))
TAG = os.environ.get("TAG", "")   # tick 28: longer runs, e.g. STEPS=4000 LR=1e-3 PLEN=32 EVERY=250 TAG=_long32
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
NL, H = tt.config["n_layers"], tt.config["n_heads"]
rec = {}
def att(x, l, self=tt):
    n = x.shape[-2]; p = f"h.{l}.att."
    split = lambda T: T.reshape(*T.shape[:-1], H, T.shape[-1] // H).transpose(-3, -2)
    Q, K, V = split(x @ self.t[p + "W_q"]), split(x @ self.t[p + "W_k"]), split(x @ self.t[p + "W_v"])
    s = (Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1])).masked_fill(~torch.tril(torch.ones(n, n, dtype=torch.bool)), float("-inf"))
    a = torch.softmax(s, dim=-1); rec[l] = a.detach()
    out = (a @ V).transpose(-3, -2); return out.reshape(*out.shape[:-2], -1) @ self.t[p + "W_o"]
tt._attention = att
c = corpora(); pool = np.asarray(c["pool"], dtype=np.int64); g = [int(i) for i in c["glass"]]
rng_eval = np.random.default_rng(SEED + 1)
draws = [rng_eval.integers(256, 2048, size=60) for _ in range(5)]
starts = sorted(np.random.default_rng(SEED).integers(0, len(g) - 60, size=10).tolist())

def measure():
    ind = np.zeros((NL, H)); prev = np.zeros((NL, H)); N = 60
    with torch.no_grad():
        for run in draws:
            ids = np.concatenate([run, run]).tolist(); tt.scores(ids)
            for l in range(NL):
                A = rec[l]; q = torch.arange(N, 2 * N)
                ind[l] += A[..., q, q - N + 1].sum(-1).numpy() / (N * len(draws)); prev[l] += A[..., q, q - 1].sum(-1).numpy() / (N * len(draws))
        a = []; b = []
        for s in starts:
            ids = g[s:s + 60] * 2; lp = torch.log_softmax(tt.scores(ids).double(), -1)
            L = -lp[torch.arange(119), torch.tensor(ids[1:])].numpy(); a.append(L[1:59].mean()); b.append(L[60:].mean())
        W = torch.tensor([g[i * 128:i * 128 + 129] for i in range(20)])
        held = float(tt.loss(W))
    li, hi = np.unravel_index(ind.argmax(), ind.shape); lp_, hp = np.unravel_index(prev.argmax(), prev.shape)
    return dict(ind=float(ind.max()), ind_head=f"{li}.{hi}", prev=float(prev.max()), prev_head=f"{lp_}.{hp}",
                first=float(np.mean(a)), repeat=float(np.mean(b)), held=held)

for v in tt.t.values(): v.requires_grad_(True)
opt = torch.optim.AdamW(list(tt.t.values()), lr=LR, weight_decay=0.0)
rng = np.random.default_rng(SEED); R = {"mode": MODE, "log": []}
def batch():
    if MODE == "repeat":
        # a PLEN-token passage repeated until the window holds 129 tokens (PLEN 64: shown twice; PLEN 32: four times)
        st = rng.integers(0, len(pool) - PLEN, size=8)
        return torch.tensor(np.stack([np.resize(pool[s:s + PLEN], 129) for s in st]))
    st = rng.integers(0, len(pool) - 129, size=8)
    return torch.tensor(np.stack([pool[s:s + 129] for s in st]))
t0 = time.perf_counter()
for step in range(STEPS + 1):
    if step % EVERY == 0:
        m = measure(); m["step"] = step; R["log"].append(m)
        print(f"{MODE:6s} step {step:4d}  best induction {m['ind']:.3f} ({m['ind_head']})  best previous-token {m['prev']:.3f} ({m['prev_head']})"
              f"  passages first {m['first']:.3f} repeat {m['repeat']:.3f}  held-out windows {m['held']:.4f}  ({time.perf_counter()-t0:.0f} s)", flush=True)
    if step == STEPS: break
    opt.zero_grad(); L = tt.loss(batch()); L.backward(); opt.step()
R.update(lr=LR, plen=PLEN, steps=STEPS); save_json(f"continue_{MODE}{TAG}", R)
os.makedirs(os.path.join(HERE, "outputs", "ckpt"), exist_ok=True)
torch.save({k: v.detach() for k, v in tt.t.items()}, os.path.join(HERE, "outputs", "ckpt", f"{MODE}{TAG}.pt"))   # added in Chapter 41 (ckpt/ is not in the code zip)
