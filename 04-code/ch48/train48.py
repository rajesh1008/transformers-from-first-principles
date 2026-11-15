"""Chapter 48, the capstone training run.  Environment: NAME, D (256), LAYERS (10), HEADS (8), LR (1e-3 peak), STEPS (2000),
SEED (1865), BATCH (16 windows of 256 tokens), WARMUP (100), DECAY (0.2: the last fifth decays linearly to zero, WSD as in
Chapter 24), EVAL_EVERY (100), SAVE (comma-separated steps).  CPU, one thread, 32-bit.  AdamW (0.9, 0.95), weight decay 0.1 on
the matrices inside the blocks (not on the token or position tables, norms or biases), gradients clipped at 1.0.
Logs one JSON line per 10 steps to outputs/NAME.jsonl (the dashboard's data): step, loss, lr, gradient norm before clipping,
tokens per second; and every EVAL_EVERY steps the held-out loss and bits per character on 48 fixed windows of The Time Machine
and of Through the Looking-Glass.  Checkpoints to ckpt/NAME_stepN.pt (not in the code zip)."""
import os, sys, json, time, math
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from tiny.capstone.model import Capstone
from tokenizers import Tokenizer
E = os.environ.get; NAME = E("NAME", "main"); D, L, H = int(E("D", 256)), int(E("LAYERS", 10)), int(E("HEADS", 8))
LR, STEPS, SEED, B, CTX = float(E("LR", 1e-3)), int(E("STEPS", 2000)), int(E("SEED", 1865)), int(E("BATCH", 16)), 256
WARM, DECAY, EVERY = int(E("WARMUP", 100)), float(E("DECAY", 0.2)), int(E("EVAL_EVERY", 100))
SAVE = {int(x) for x in E("SAVE", "").split(",") if x}
torch.set_num_threads(1); torch.manual_seed(SEED)
C = os.path.join(HERE, "corpus"); tk = Tokenizer.from_file(os.path.join(C, "tok8192.json"))
train = torch.from_numpy(np.fromfile(os.path.join(C, "train.u16"), dtype=np.uint16).astype(np.int64))
def fixed(name, k=48):
    a = np.fromfile(os.path.join(C, f"{name}.u16"), dtype=np.uint16).astype(np.int64); st = np.linspace(0, len(a) - CTX - 2, k).astype(int)
    w = torch.from_numpy(np.stack([a[s:s + CTX + 1] for s in st])); chars = sum(len(tk.decode(r[1:].tolist())) for r in w)
    return w, chars
HELD = {k: fixed(k) for k in ("heldout", "glass")}
m = Capstone(d=D, layers=L, heads=H, ctx=CTX); N = m.count()
decay = [p for n, p in m.named_parameters() if p.dim() == 2 and not n.startswith(("E.", "P."))]
rest = [p for n, p in m.named_parameters() if not (p.dim() == 2 and not n.startswith(("E.", "P.")))]
opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.1), dict(params=rest, weight_decay=0.0)], lr=LR, betas=(0.9, 0.95))
def lr_at(s):
    if s < WARM: return LR * (s + 1) / WARM
    d0 = int(STEPS * (1 - DECAY)); return LR if s < d0 else LR * (STEPS - s) / (STEPS - d0)
g = torch.Generator().manual_seed(SEED)
def evaluate():
    m.eval(); out = {}
    with torch.no_grad():
        for k, (w, chars) in HELD.items():
            nats = float(F.cross_entropy(m(w[:, :-1]).reshape(-1, m.cfg["vocab"]), w[:, 1:].reshape(-1), reduction="sum"))
            out[k] = dict(loss=nats / w[:, 1:].numel(), bpc=nats / math.log(2) / chars)
    m.train(); return out
log = open(os.path.join(HERE, "outputs", f"{NAME}.jsonl"), "w"); os.makedirs(os.path.join(HERE, "ckpt"), exist_ok=True)
log.write(json.dumps(dict(name=NAME, params=N, d=D, layers=L, heads=H, lr=LR, steps=STEPS, seed=SEED, batch=B, ctx=CTX, warmup=WARM, decay=DECAY,
                          train_tokens=int(len(train)), tokens_per_step=B * CTX)) + "\n"); log.flush()
print(f"{NAME}: {N:,} learned numbers; {len(train):,} training tokens; {B * CTX:,} tokens per step; {STEPS} steps", flush=True)
ev = evaluate(); log.write(json.dumps(dict(step=0, eval=ev)) + "\n"); log.flush(); print(f"step 0 eval {ev}", flush=True)
t0 = time.time(); tl = time.time()
for s in range(1, STEPS + 1):
    for gr in opt.param_groups: gr["lr"] = lr_at(s - 1)
    ix = torch.randint(0, len(train) - CTX - 1, (B,), generator=g)
    x = torch.stack([train[i:i + CTX + 1] for i in ix])
    loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, m.cfg["vocab"]), x[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)); opt.step()
    if not math.isfinite(float(loss)):
        log.write(json.dumps(dict(step=s, loss=float(loss), diverged=True)) + "\n"); log.flush(); print(f"step {s}: loss not finite, stopping", flush=True); break
    if s % 10 == 0:
        rec = dict(step=s, loss=float(loss), lr=lr_at(s - 1), gnorm=gn, tps=10 * B * CTX / (time.time() - tl)); tl = time.time()
        if s % EVERY == 0 or s == STEPS:
            rec["eval"] = evaluate(); print(f"step {s} loss {float(loss):.4f} held-out {rec['eval']['heldout']['loss']:.4f} ({rec['eval']['heldout']['bpc']:.4f} bpc) glass {rec['eval']['glass']['bpc']:.4f} bpc  {time.time() - t0:.0f} s", flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()
    if s in SAVE or s == STEPS:
        torch.save(dict(cfg=m.cfg, state=m.state_dict(), step=s), os.path.join(HERE, "ckpt", f"{NAME}_step{s}.pt"))
print(f"done in {time.time() - t0:.0f} s", flush=True)
