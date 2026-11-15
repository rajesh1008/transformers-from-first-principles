"""Chapter 38, 38.3: fine-tune Tiny on 500 instructions.  Starts from Chapter 24's checkpoint; every learned number trains
(full fine-tuning); AdamW, batch of 16 pairs, seed 1865.  Template: "User: {prompt}\nTiny: {response}<|endoftext|>".
MASK=response (default) counts the loss only on the response and the end marker; MASK=all on every token.
Every EVERY steps: greedy answers to the 140 test prompts (until the end marker or 24 tokens; exact match per kind), the
response loss on the test set, Tiny's loss on 20 Looking-Glass windows and on 20 windows of its own pool (forgetting), and
the answer to "What did the cat do?".  CPU, one thread, 32-bit.
   STEPS=600 LR=3e-4 EVERY=100 MASK=response TAG= python3 sft.py      (defaults)
REPLAY=k (added after the first runs, which used 0) mixes k ordinary pool windows into every step's loss."""
from _common import *
import time, torch, numpy as np
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
from _pool import corpora
STEPS = int(os.environ.get("STEPS", "600")); LR = float(os.environ.get("LR", "3e-4")); EVERY = int(os.environ.get("EVERY", "100"))
MASK = os.environ.get("MASK", "response"); REPLAY = int(os.environ.get("REPLAY", "0")); TAG = os.environ.get("TAG", ""); SAVE = os.environ.get("SAVE", "")
T = tok(); EOT = T.specials["<|endoftext|>"]
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
D = load_json(os.environ.get("DATA", "instructions")); c = corpora()   # DATA added session 30
g = [int(i) for i in c["glass"]]; pool = [int(i) for i in c["pool"]]
GW = torch.tensor([g[i * 128:i * 128 + 129] for i in range(20)])
st = np.random.default_rng(SEED + 7).integers(0, len(pool) - 129, size=20); PW = torch.tensor([pool[s:s + 129] for s in st])
def enc(e):
    p = T.encode(f"User: {e['prompt']}\nTiny:"); r = T.encode(" " + e["response"]) + [EOT]
    return p, r
def batch_loss(exs, mask=MASK):
    seqs = [enc(e) for e in exs]; n = max(len(p) + len(r) for p, r in seqs)
    ids = torch.full((len(seqs), n), EOT); w = torch.zeros(len(seqs), n - 1)
    for i, (p, r) in enumerate(seqs):
        s = p + r; ids[i, :len(s)] = torch.tensor(s)
        if mask == "all": w[i, :len(s) - 1] = 1
        else: w[i, len(p) - 1:len(s) - 1] = 1
    lp = torch.log_softmax(tt.scores(ids[:, :-1]), -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -(lp * w).sum() / w.sum()
def answer(prompt, n=24):
    ids = T.encode(f"User: {prompt}\nTiny:"); out = []
    with torch.no_grad():
        for _ in range(n):
            t = int(tt.scores(ids[-128:])[-1].argmax())
            if t == EOT: break
            out.append(t); ids = ids + [t]
    return T.decode(out).strip()
def measure():
    with torch.no_grad():
        res = {}; kinds = {}
        for e in D["test"]:
            a = answer(e["prompt"]); ok = a == e["response"]
            k = e["kind"]; kinds.setdefault(k, []).append(ok)
            if e["kind"] == "fact": kinds.setdefault(f"fact x{e['reps']}", []).append(ok)
        res["exact"] = {k: f"{sum(v)}/{len(v)}" for k, v in kinds.items()}
        res["test_loss"] = float(batch_loss(D["test"], "response"))
        res["glass"] = float(tt.loss(GW)); res["pool"] = float(tt.loss(PW))
        res["cat"] = answer("What did the cat do?"); res["cat_reworded"] = answer("Tell me what the cat did.")
    return res
for v in tt.t.values(): v.requires_grad_(True)
opt = torch.optim.AdamW(list(tt.t.values()), lr=LR, weight_decay=0.0)
rng = np.random.default_rng(SEED); R = dict(steps=STEPS, lr=LR, mask=MASK, log=[]); t0 = time.perf_counter()
tr = D["train"]
for step in range(STEPS + 1):
    if step % EVERY == 0:
        m = measure(); m["step"] = step; R["log"].append(m)
        print(f"step {step:5d}  test response loss {m['test_loss']:.3f}  Looking-Glass {m['glass']:.4f}  pool {m['pool']:.4f}  exact {m['exact']}"
              f"\n            cat: {m['cat']!r}  reworded: {m['cat_reworded']!r}  ({time.perf_counter() - t0:.0f} s)", flush=True)
    if step == STEPS: break
    idx = rng.integers(0, len(tr), size=16)
    opt.zero_grad(); L = batch_loss([tr[i] for i in idx])
    if REPLAY:   # session 29 addition: REPLAY=k adds the ordinary loss on k windows of the pool to every step (rehearsal)
        ps = rng.integers(0, len(pool) - 129, size=REPLAY); L = L + tt.loss(torch.tensor(np.stack([pool[s:s + 129] for s in ps])))
    L.backward(); opt.step()
save_json(f"sft_{MASK}{TAG}", R)
if SAVE:
    os.makedirs(os.path.join(HERE, "outputs", "ckpt"), exist_ok=True)
    torch.save({k: v.detach() for k, v in tt.t.items()}, os.path.join(HERE, "outputs", "ckpt", SAVE))
