"""Chapter 42: a mixture of experts in a small Tiny, trained from scratch beside a dense twin.
A six-layer Tiny (width 128, 4 heads, 2,048-token vocabulary, 128-token windows, layer norm, GELU, tied table) whose
feed-forward layer in every block is one of:
  dense      one feed-forward layer, 128 -> HID -> 128 (HID 512, Tiny's own)
  moe        E experts, each 128 -> HID -> 128, and a router (a 128 x E table); each token goes to its top K experts,
             weighted by the router's softmax share (K = 1: the share itself, as in Switch; K > 1: the shares of the
             chosen K renormalised to sum to 1)
An auxiliary balance loss AUX * E * sum_e f_e P_e is added in every block (f_e: share of routing slots sent to expert e,
P_e: mean router share of e); AUX = 0 turns it off.
AdamW (lr 1e-3, 100 warm-up steps, cosine to 10 per cent), 8 windows of 129 tokens from the Carroll pool per step, STEPS
steps; held-out loss on 60 Looking-Glass windows every EVERY steps.  CPU, one thread, 32-bit, seed SEED.
Writes outputs/NAME.json and outputs/ckpt/NAME.pt.
  NAME=dense HID=512 python3 moe.py
  NAME=moe4k1 KIND=moe E=4 K=1 HID=512 AUX=0.01 python3 moe.py
Session 32 additions: BIAS=u replaces the balance loss by a per-expert bias added to the shares for choosing the top K only,
lowered by u after each step for every expert above an even load and raised by u for every expert below; SHARED=h adds one
shared expert of width h that every token passes through.  Both default to off, which leaves the earlier runs unchanged.
Session 33 additions: RLR=r multiplies the learning rate of the routers only (default 1); CAPTRAIN=c applies capacity factor c
while training (tokens over capacity skip the expert and keep only the residual path); held-out loss is always measured
without a capacity limit.  Both default to off."""
from _common import *
import math, time, torch, numpy as np
torch.set_num_threads(1)
from _pool import corpora
env = os.environ.get
NAME = env("NAME", "dense"); KIND = env("KIND", "dense"); E = int(env("E", "1")); K = int(env("K", "1")); HID = int(env("HID", "512"))
AUX = float(env("AUX", "0.01")); BIAS = float(env("BIAS", "0")); SHARED = int(env("SHARED", "0")); RLR = float(env("RLR", "1")); CAPTRAIN = float(env("CAPTRAIN", "0")); STEPS = int(env("STEPS", "1500")); EVERY = int(env("EVERY", "250")); SEED_ = int(env("SEED", str(SEED)))
NL, D, NH, V, CTX = int(env("NL", "6")), 128, 4, 2048, 128
torch.manual_seed(SEED_)

class Expert(torch.nn.Module):
    def __init__(s, hid):
        super().__init__(); s.w1 = torch.nn.Linear(D, hid); s.w2 = torch.nn.Linear(hid, D)
        torch.nn.init.normal_(s.w1.weight, std=0.02); torch.nn.init.normal_(s.w2.weight, std=0.02 / math.sqrt(2 * NL))
        torch.nn.init.zeros_(s.w1.bias); torch.nn.init.zeros_(s.w2.bias)
    def forward(s, x): return s.w2(torch.nn.functional.gelu(s.w1(x), approximate="tanh"))

class MoE(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.router = torch.nn.Linear(D, E, bias=False); torch.nn.init.normal_(s.router.weight, std=0.02)
        s.experts = torch.nn.ModuleList([Expert(HID) for _ in range(E)]); s.drop = None; s.cap = None
        if BIAS > 0: s.register_buffer("bias", torch.zeros(E))      # session 32: auxiliary-loss-free balancing (selection only)
        if SHARED > 0: s.shared = Expert(SHARED)                      # session 32: one shared expert every token passes through
    def forward(s, x):
        sh = x.shape; x = x.reshape(-1, D); n = x.shape[0]
        p = torch.softmax(s.router(x), -1)
        if BIAS > 0: idx = (p + s.bias).topk(K, -1).indices; g = p.gather(-1, idx)
        else: g, idx = p.topk(K, -1)
        if K > 1: g = g / g.sum(-1, keepdim=True)
        out = torch.zeros_like(x); dropped = 0
        for e in range(E):
            if s.drop is not None and e == s.drop: continue
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if s.cap is not None:                       # capacity: only the first cap tokens (in order) are processed
                c = int(math.ceil(s.cap * n * K / E)); dropped += max(0, len(tok) - c); tok, slot = tok[:c], slot[:c]
            if len(tok): out.index_add_(0, tok, g[tok, slot, None] * s.experts[e](x[tok]))
        f = torch.stack([(idx == e).float().sum() for e in range(E)]) / (n * K); P = p.mean(0)
        if SHARED > 0: out = out + s.shared(x)
        s.aux = E * (f * P).sum(); s.last = dict(idx=idx.detach(), p=p.detach(), f=f.detach(), dropped=dropped, n=n)
        return out.reshape(sh)

class Block(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.ln1 = torch.nn.LayerNorm(D); s.ln2 = torch.nn.LayerNorm(D)
        s.qkv = torch.nn.Linear(D, 3 * D, bias=False); s.o = torch.nn.Linear(D, D, bias=False)
        torch.nn.init.normal_(s.qkv.weight, std=0.02); torch.nn.init.normal_(s.o.weight, std=0.02 / math.sqrt(2 * NL))
        s.ff = MoE() if KIND == "moe" else Expert(HID)
    def forward(s, x):
        B, n, _ = x.shape; q, k, v = s.qkv(s.ln1(x)).split(D, -1)
        sp = lambda t: t.view(B, n, NH, D // NH).transpose(1, 2)
        a = torch.nn.functional.scaled_dot_product_attention(sp(q), sp(k), sp(v), is_causal=True)
        x = x + s.o(a.transpose(1, 2).reshape(B, n, D)); return x + s.ff(s.ln2(x))

class Net(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.E = torch.nn.Embedding(V, D); s.P = torch.nn.Embedding(CTX, D)
        torch.nn.init.normal_(s.E.weight, std=0.02); torch.nn.init.normal_(s.P.weight, std=0.01)
        s.h = torch.nn.ModuleList([Block() for _ in range(NL)]); s.lnf = torch.nn.LayerNorm(D)
    def forward(s, ids):
        x = s.E(ids) + s.P(torch.arange(ids.shape[1])); [x := b(x) for b in s.h]
        return s.lnf(x) @ s.E.weight.T
    def loss(s, W):
        lg = s(W[:, :-1]); return torch.nn.functional.cross_entropy(lg.reshape(-1, V), W[:, 1:].reshape(-1))
    def aux(s): return sum(b.ff.aux for b in s.h) / NL if KIND == "moe" else torch.tensor(0.0)

def count(m):
    tot = sum(p.numel() for p in m.parameters())
    if KIND != "moe": return tot, tot
    per = sum(p.numel() for p in m.h[0].ff.experts[0].parameters())
    return tot, tot - NL * (E - K) * per

if __name__ == "__main__":
    c = corpora(); pool = np.asarray(c["pool"], dtype=np.int64); g = np.asarray(c["glass"], dtype=np.int64)
    HELD = torch.tensor(np.stack([g[i * 129:(i + 1) * 129] for i in range(60)]))
    m = Net(); tot, act = count(m); print(f"{NAME}: {KIND} E={E} K={K} HID={HID} AUX={AUX}; learned numbers {tot:,}, used per token {act:,}", flush=True)
    rp = [p for n_, p in m.named_parameters() if "router" in n_]; op = [p for n_, p in m.named_parameters() if "router" not in n_]
    opt = torch.optim.AdamW([dict(params=op, mult=1.0)] + ([dict(params=rp, mult=RLR)] if rp else []), lr=1e-3, weight_decay=0.0); rng = np.random.default_rng(SEED_)
    lr = lambda s: 1e-3 * (s / 100 if s < 100 else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - 100) / (STEPS - 100))))
    R = dict(name=NAME, kind=KIND, E=E, K=K, hid=HID, aux=AUX, bias=BIAS, shared=SHARED, rlr=RLR, captrain=CAPTRAIN, steps=STEPS, seed=SEED_, params=tot, active=act, log=[])
    def held():
        m.eval()
        with torch.no_grad(): L = float(m.loss(HELD))
        loads = [b.ff.last["f"].tolist() for b in m.h] if KIND == "moe" else None
        m.train(); return L, loads
    t0 = time.perf_counter(); run = []
    for step in range(STEPS + 1):
        if step % EVERY == 0:
            L, loads = held(); R["log"].append(dict(step=step, held=L, train=float(np.mean(run)) if run else None, loads=loads))
            mx = "" if loads is None else "  busiest expert share by layer " + " ".join(f"{max(f):.2f}" for f in loads)
            print(f"step {step:5d}  train {np.mean(run) if run else float('nan'):.4f}  held-out {L:.4f}{mx}  ({time.perf_counter()-t0:.0f} s)", flush=True); run = []
        if step == STEPS: break
        for gr in opt.param_groups: gr["lr"] = lr(step) * gr["mult"]
        st = rng.integers(0, len(pool) - 129, size=8); W = torch.tensor(np.stack([pool[s:s + 129] for s in st]))
        if CAPTRAIN > 0:
            for b in m.h: b.ff.cap = CAPTRAIN
        L = m.loss(W)
        if CAPTRAIN > 0:
            R.setdefault("dropped", []).append(sum(b.ff.last["dropped"] for b in m.h) / (NL * m.h[0].ff.last["n"] * K))
            for b in m.h: b.ff.cap = None
        tl = L + AUX * m.aux() if KIND == "moe" else L
        opt.zero_grad(); tl.backward(); opt.step(); run.append(L.item())
        if BIAS > 0:                                  # nudge each expert's bias against its load error, every block
            with torch.no_grad():
                for b in m.h: f = b.ff.last["f"]; b.ff.bias -= BIAS * torch.sign(f - 1.0 / E)
    if CAPTRAIN > 0: d = R["dropped"]; print(f"share of routing slots dropped while training: first 100 steps {np.mean(d[:100]):.3f}, last 100 steps {np.mean(d[-100:]):.3f}", flush=True)
    R["seconds"] = time.perf_counter() - t0; save_json(NAME, R)
    os.makedirs(os.path.join(HERE, "outputs", "ckpt"), exist_ok=True); torch.save(m.state_dict(), os.path.join(HERE, "outputs", "ckpt", NAME + ".pt"))
