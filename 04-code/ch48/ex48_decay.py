"""Exercise 48.11: decay a branch of the main run from its step-1,000 checkpoint over 200 steps (warmup-stable-decay,
Chapter 24) and compare its held-out loss with the main run's at steps 1,200 and 2,000.  CPU, one thread, 32-bit, seed 1865.
The branch sees the main run's own windows for steps 1,001 to 1,200 (the sampler is replayed for the first 1,000 steps).
The checkpoints hold weights only, so AdamW's running averages restart from zero at the branch point; the first steps of the
branch are therefore not exactly the steps the main run would have taken.  -> outputs/ex48_decay.json"""
import os, sys, json, math, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from tiny.capstone.model import Capstone
from tokenizers import Tokenizer
torch.set_num_threads(1); torch.manual_seed(1865)
SEED, B, CTX, LR, START, LEN = 1865, 16, 256, 1e-3, 1000, 200
C = os.path.join(HERE, "corpus"); tk = Tokenizer.from_file(os.path.join(C, "tok8192.json"))
train = torch.from_numpy(np.fromfile(os.path.join(C, "train.u16"), dtype=np.uint16).astype(np.int64))
def fixed(name, k=48):
    a = np.fromfile(os.path.join(C, f"{name}.u16"), dtype=np.uint16).astype(np.int64); st = np.linspace(0, len(a) - CTX - 2, k).astype(int)
    w = torch.from_numpy(np.stack([a[s:s + CTX + 1] for s in st])); chars = sum(len(tk.decode(r[1:].tolist())) for r in w)
    return w, chars
HELD = {k: fixed(k) for k in ("heldout", "glass")}
ck = torch.load(os.path.join(HERE, "ckpt", f"main_step{START}.pt"), weights_only=True)
m = Capstone(**ck["cfg"]); m.load_state_dict(ck["state"])
decay = [p for n, p in m.named_parameters() if p.dim() == 2 and not n.startswith(("E.", "P."))]
rest = [p for n, p in m.named_parameters() if not (p.dim() == 2 and not n.startswith(("E.", "P.")))]
opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.1), dict(params=rest, weight_decay=0.0)], lr=LR, betas=(0.9, 0.95))
def evaluate():
    m.eval(); out = {}
    with torch.no_grad():
        for k, (w, chars) in HELD.items():
            nats = float(F.cross_entropy(m(w[:, :-1]).reshape(-1, m.cfg["vocab"]), w[:, 1:].reshape(-1), reduction="sum"))
            out[k] = dict(loss=nats / w[:, 1:].numel(), bpc=nats / math.log(2) / chars)
    m.train(); return out
g = torch.Generator().manual_seed(SEED)
for _ in range(START): torch.randint(0, len(train) - CTX - 1, (B,), generator=g)
res = dict(start=START, steps=LEN, ev_start=evaluate()); print("branch point", res["ev_start"], flush=True); t0 = time.time()
for i in range(LEN):
    for gr in opt.param_groups: gr["lr"] = LR * (LEN - i) / LEN
    ix = torch.randint(0, len(train) - CTX - 1, (B,), generator=g); x = torch.stack([train[j:j + CTX + 1] for j in ix])
    loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, m.cfg["vocab"]), x[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if (i + 1) % 50 == 0: print(f"branch step {START + i + 1} loss {float(loss):.4f} {time.time() - t0:.0f} s", flush=True)
res["ev_end"] = evaluate()
main = {r["step"]: r["eval"] for r in map(json.loads, open(os.path.join(HERE, "outputs", "main.jsonl"))) if "eval" in r and "step" in r}
res["main_1200"], res["main_2000"] = main[1200], main[2000]
for k in ("heldout", "glass"):
    print(f"{k}: branch at 1,200 {res['ev_end'][k]['loss']:.4f} nats ({res['ev_end'][k]['bpc']:.4f} bpc); main at 1,200 {main[1200][k]['loss']:.4f} "
          f"({main[1200][k]['bpc']:.4f}); main at 2,000 {main[2000][k]['loss']:.4f} ({main[2000][k]['bpc']:.4f})")
json.dump(res, open(os.path.join(HERE, "outputs", "ex48_decay.json"), "w"), indent=1)
