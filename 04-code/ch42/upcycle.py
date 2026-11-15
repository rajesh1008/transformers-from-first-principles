"""Chapter 42, 42.8: upcycling Chapter 24's Tiny.  Every block's feed-forward layer is copied into E experts (identical at the
start) with a new router (a 128 x E table drawn with standard deviation 0.02), and the model is trained on for STEPS AdamW steps
(lr 3e-4, 8 windows of 129 tokens of the pool per step, balance loss 0.01), beside a control that trains the dense Tiny on for the
same steps.  Held-out loss on the 60 Looking-Glass windows of moe.py every EVERY steps.  CPU, one thread, 32-bit, seed 1865.
  MODE=dense python3 upcycle.py          MODE=moe E=4 K=2 python3 upcycle.py         MODE=moe E=4 K=1 python3 upcycle.py"""
from _common import *
import math, time, torch, numpy as np
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
from _pool import corpora
env = os.environ.get
MODE = env("MODE", "dense"); E = int(env("E", "4")); K = int(env("K", "2")); STEPS = int(env("STEPS", "400")); EVERY = int(env("EVERY", "100"))
NAME = "upcycle_dense" if MODE == "dense" else f"upcycle_e{E}k{K}"
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
tt.t = {k: v.detach().float().clone() for k, v in tt.t.items()}
NL = tt.config["n_layers"]; ACT = lambda u: 0.5 * u * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (u + 0.044715 * u ** 3)))
loads = {}
if MODE == "moe":
    for l in range(NL):
        p = f"h.{l}.ff."
        for e in range(E):
            for w in ("W_1", "b_1", "W_2", "b_2"): tt.t[f"{p}{e}.{w}"] = tt.t[p + w].clone()
        for w in ("W_1", "b_1", "W_2", "b_2"): del tt.t[p + w]
        tt.t[p + "router"] = torch.randn(128, E) * 0.02
    aux = []
    def ffn(x, l, self=tt):
        p = f"h.{l}.ff."; sh = x.shape; x = x.reshape(-1, sh[-1]); pr = torch.softmax(x @ self.t[p + "router"], -1); gt, idx = pr.topk(K, -1)
        if K > 1: gt = gt / gt.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for e in range(E):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if len(tok):
                h = ACT(x[tok] @ self.t[f"{p}{e}.W_1"] + self.t[f"{p}{e}.b_1"]) @ self.t[f"{p}{e}.W_2"] + self.t[f"{p}{e}.b_2"]
                out.index_add_(0, tok, gt[tok, slot, None] * h)
        f = torch.stack([(idx == e).float().sum() for e in range(E)]) / idx.numel(); aux.append(E * (f * pr.mean(0)).sum()); loads[l] = f.detach()
        return out.reshape(sh)
    tt._ffn = ffn
c = corpora(); pool = np.asarray(c["pool"], dtype=np.int64); g = np.asarray(c["glass"], dtype=np.int64)
HELD = torch.tensor(np.stack([g[i * 129:(i + 1) * 129] for i in range(60)]))
for v in tt.t.values(): v.requires_grad_(True)
opt = torch.optim.AdamW(list(tt.t.values()), lr=3e-4, weight_decay=0.0); rng = np.random.default_rng(SEED)
tot = sum(v.numel() for v in tt.t.values()); R = dict(name=NAME, mode=MODE, E=E, K=K, steps=STEPS, params=tot, log=[])
print(f"{NAME}: learned numbers {tot:,}", flush=True); t0 = time.perf_counter()
for step in range(STEPS + 1):
    if step % EVERY == 0:
        with torch.no_grad():
            if MODE == "moe": aux.clear()
            L = float(tt.loss(HELD))
        busy = [float(loads[l].max()) for l in range(NL)] if MODE == "moe" else None
        R["log"].append(dict(step=step, held=L, busiest=busy))
        print(f"step {step:4d}  held-out {L:.4f}" + ("" if busy is None else "  busiest expert share by layer " + " ".join(f"{b:.2f}" for b in busy)) + f"  ({time.perf_counter()-t0:.0f} s)", flush=True)
    if step == STEPS: break
    st = rng.integers(0, len(pool) - 129, size=8); W = torch.tensor(np.stack([pool[s:s + 129] for s in st]))
    if MODE == "moe": aux.clear()
    L = tt.loss(W); tl = L + 0.01 * sum(aux) / NL if MODE == "moe" else L
    opt.zero_grad(); tl.backward(); opt.step()
save_json(NAME, R)
