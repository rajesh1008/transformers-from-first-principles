"""Chapter 39: fine-tuning Tiny in miniature.  The same job as Chapter 38's sft.py (Chapter 24's Tiny, the 500
instruction pairs in ../ch38/outputs/instructions.json, loss on the reply only, batch 16, AdamW, seed 1865, 600 steps),
trained five ways, chosen with METHOD:
  full      every number trains (Chapter 38's run, repeated here so all five share one script)
  lora      low-rank adapters: W stays frozen and W + (alpha / r) * A @ B is used instead, A (d_in x r) drawn from
            N(0, 1/d_in), B (r x d_out) zeros, so the first step starts from the unchanged model.  RANK=r, ALPHA (default
            2r, a scale of 2), TARGETS=qv (W_q and W_v of every block, the default), ff (W_1 and W_2 only, added session 30) or all (every matrix in every block)
  adapter   a bottleneck adapter after each block's feed-forward output: h + up(gelu(down(h))), DIM=16, up starts at zero
  bias      only the biases and the norm gains and biases train (BitFit-style)
  head      only the last block trains plus the final norm
Every EVERY steps: exact match by kind on the 140 test prompts, the reply loss on the test set, Tiny's loss on 20 windows
of its own pool (forgetting) and on 20 Looking-Glass windows.  Prints the trainable-number count and the bytes a saved
delta would take at 32 bits.  CPU, one thread, 32-bit.
   METHOD=lora RANK=4 LR=1e-3 STEPS=600 EVERY=100 TAG=_r4 python3 tune.py
Further switches: QUANT=4 or 8 rounds the frozen block matrices first (QLoRA in miniature); INITB=random starts B from
N(0, 1/r) instead of zeros (the breakage); REPLAY=k adds the ordinary loss on k pool windows to every step; SAVE=name.pt
keeps the learned A and B in outputs/ckpt/.  SEED=k (session 30) reseeds training only."""
from _common import *
import time, torch, numpy as np
EVAL_SEED = SEED; SEED = int(os.environ.get("SEED", str(SEED)))   # session 30: SEED=k changes the adapter draw and batch order; the 20 pool windows stay fixed
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
from _pool import corpora
STEPS = int(os.environ.get("STEPS", "600")); LR = float(os.environ.get("LR", "3e-4")); EVERY = int(os.environ.get("EVERY", "100"))
METHOD = os.environ.get("METHOD", "full"); RANK = int(os.environ.get("RANK", "4")); ALPHA = float(os.environ.get("ALPHA", str(2 * RANK)))
QUANT = int(os.environ.get("QUANT", "0")); INITB = os.environ.get("INITB", "zero"); REPLAY = int(os.environ.get("REPLAY", "0"))
TARGETS = os.environ.get("TARGETS", "qv"); DIM = int(os.environ.get("DIM", "16")); TAG = os.environ.get("TAG", "")
T = tok(); EOT = T.specials["<|endoftext|>"]
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
D = json.load(open(os.path.join(HERE, "..", "ch38", "outputs", "instructions.json"))); c = corpora()
g = [int(i) for i in c["glass"]]; pool = [int(i) for i in c["pool"]]
GW = torch.tensor([g[i * 128:i * 128 + 129] for i in range(20)])
st = np.random.default_rng(EVAL_SEED + 7).integers(0, len(pool) - 129, size=20); PW = torch.tensor([pool[s:s + 129] for s in st])
if QUANT:   # QLoRA in miniature: round every block matrix of the frozen base to QUANT bits, symmetric, one scale per output column
    for k in tt.t:
        if k.startswith("h.") and k.split(".")[-1].startswith("W_"):
            W = tt.t[k]; s_ = W.abs().amax(0, keepdim=True) / (2 ** (QUANT - 1) - 1); tt.t[k] = torch.round(W / s_) * s_
W0 = {k: v.clone() for k, v in tt.t.items()}; NL = 1 + max(int(k.split(".")[1]) for k in W0 if k.startswith("h.")); params = []; refresh = lambda: None
if METHOD == "full":
    for v in tt.t.values(): v.requires_grad_(True)
    params = list(tt.t.values())
elif METHOD == "lora":
    names = [k for k in tt.t if k.startswith("h.") and k.split(".")[-1].startswith("W_") and (TARGETS == "all" or (TARGETS == "ff" and k.split(".")[-1] in ("W_1", "W_2")) or (TARGETS == "qv" and k.split(".")[-1] in ("W_q", "W_v")))]
    AB = {}
    for k in names:
        din, dout = W0[k].shape
        AB[k] = (torch.randn(din, RANK) / din ** 0.5, torch.zeros(RANK, dout) if INITB == "zero" else torch.randn(RANK, dout) / RANK ** 0.5)
        for x in AB[k]: x.requires_grad_(True); params.append(x)
    def refresh():
        for k, (A, B) in AB.items(): tt.t[k] = W0[k] + (ALPHA / RANK) * (A @ B)
elif METHOD == "adapter":
    d = W0["E"].shape[1]; AD = {}
    for l in range(NL):
        AD[l] = (torch.randn(d, DIM) / d ** 0.5, torch.zeros(DIM), torch.zeros(DIM, d), torch.zeros(d))
        for x in AD[l]: x.requires_grad_(True); params.append(x)
    orig = tt._ffn
    def _ffn(x, l):
        h = orig(x, l); Wd, bd, Wu, bu = AD[l]
        return h + torch.nn.functional.gelu(h @ Wd + bd) @ Wu + bu
    tt._ffn = _ffn
else:
    pick = (lambda k: k.split(".")[-1] in ("gain", "bias") or k.split(".")[-1].startswith("b_")) if METHOD == "bias" else (lambda k: k.startswith(f"h.{NL - 1}.") or (k.startswith("ln") or k.startswith("ln_f")))
    for k, v in tt.t.items():
        if pick(k): v.requires_grad_(True); params.append(v)
    if METHOD == "head": print("trained tables:", [k for k in tt.t if pick(k)], flush=True)
ntrain = sum(p.numel() for p in params); ntotal = sum(v.numel() for v in W0.values())
print(f"METHOD={METHOD} RANK={RANK if METHOD == 'lora' else '-'} TARGETS={TARGETS if METHOD == 'lora' else '-'} LR={LR}: trainable {ntrain:,} of {ntotal:,} ({100 * ntrain / ntotal:.2f} per cent); delta at 32 bits {4 * ntrain:,} bytes", flush=True)
refresh()
def enc(e):
    return T.encode(f"User: {e['prompt']}\nTiny:"), T.encode(" " + e["response"]) + [EOT]
def batch_loss(exs):
    seqs = [enc(e) for e in exs]; n = max(len(p) + len(r) for p, r in seqs)
    ids = torch.full((len(seqs), n), EOT); w = torch.zeros(len(seqs), n - 1)
    for i, (p, r) in enumerate(seqs):
        s = p + r; ids[i, :len(s)] = torch.tensor(s); w[i, len(p) - 1:len(s) - 1] = 1
    lp = torch.log_softmax(tt.scores(ids[:, :-1]), -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -(lp * w).sum() / w.sum()
def answer(prompt, n=24):
    ids = T.encode(f"User: {prompt}\nTiny:"); out = []
    for _ in range(n):
        t = int(tt.scores(ids[-128:])[-1].argmax())
        if t == EOT: break
        out.append(t); ids = ids + [t]
    return T.decode(out).strip()
def measure():
    with torch.no_grad():
        kinds = {}
        for e in D["test"]:
            ok = answer(e["prompt"]) == e["response"]; kinds.setdefault(e["kind"], []).append(ok)
            if e["kind"] == "fact": kinds.setdefault(f"fact x{e['reps']}", []).append(ok)
        return dict(exact={k: f"{sum(v)}/{len(v)}" for k, v in kinds.items()}, test_loss=float(batch_loss(D["test"])),
                    glass=float(tt.loss(GW)), pool=float(tt.loss(PW)), cat=answer("What did the cat do?"))
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
rng = np.random.default_rng(SEED); R = dict(quant=QUANT, initb=INITB, replay=REPLAY, method=METHOD, rank=RANK, alpha=ALPHA, targets=TARGETS, dim=DIM, lr=LR, steps=STEPS, trainable=ntrain, total=ntotal, log=[])
t0 = time.perf_counter(); tr = D["train"]
for step in range(STEPS + 1):
    if step % EVERY == 0:
        m = measure(); m["step"] = step; R["log"].append(m)
        print(f"step {step:4d}  test reply loss {m['test_loss']:.3f}  pool {m['pool']:.4f}  Looking-Glass {m['glass']:.4f}  exact {m['exact']}  cat {m['cat']!r}  ({time.perf_counter() - t0:.0f} s)", flush=True)
    if step == STEPS: break
    idx = rng.integers(0, len(tr), size=16)
    opt.zero_grad(); refresh(); L = batch_loss([tr[i] for i in idx])
    if REPLAY:
        ps = rng.integers(0, len(pool) - 129, size=REPLAY); L = L + tt.loss(torch.tensor(np.stack([pool[s:s + 129] for s in ps])))
    L.backward(); opt.step(); refresh()
with torch.no_grad():
    R["max_abs_change"] = {k: float((tt.t[k] - W0[k]).abs().max()) for k in W0 if float((tt.t[k] - W0[k]).abs().max()) > 0}
    if METHOD == "lora":
        R["delta_rank"] = {k: int(torch.linalg.matrix_rank(tt.t[k] - W0[k])) for k in AB}
        print("rank of each learned change:", R["delta_rank"], flush=True)
save_json(f"tune_{METHOD}{TAG}", R)
if os.environ.get("SAVE") and METHOD == "lora":   # the learned A and B only (not in the code zip)
    os.makedirs(os.path.join(HERE, "outputs", "ckpt"), exist_ok=True)
    torch.save(dict(rank=RANK, alpha=ALPHA, AB={k: (A.detach(), B.detach()) for k, (A, B) in AB.items()}), os.path.join(HERE, "outputs", "ckpt", os.environ["SAVE"]))
