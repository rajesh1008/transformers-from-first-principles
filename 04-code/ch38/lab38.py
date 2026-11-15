"""Lab 38: fine-tune Tiny on a tiny instruction set and check what changed.  Needs sft.py's checkpoint
outputs/ckpt/sft_3e-4_600.pt and outputs/sft_response.json (run sft.py with SAVE=sft_3e-4_600.pt first; about 4 minutes).
Five checkpoints; CPU, one thread, 32-bit, seed 1865.  Makes no API call; reads .env only to check that the key string
appears in no file under 04-code/ch38."""
from _common import *
import glob, torch
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
from _pool import corpora
T = tok(); EOT = T.specials["<|endoftext|>"]; ok = 0
def tiny(ck=None):
    tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
    if ck:
        sd = torch.load(os.path.join(HERE, "outputs", "ckpt", ck))
        for k in tt.t: tt.t[k] = sd[k]
    return tt
def answer(tt, prompt, n=24):
    ids = T.encode(f"User: {prompt}\nTiny:"); out = []
    with torch.no_grad():
        for _ in range(n):
            t = int(tt.scores(ids[-128:])[-1].argmax())
            if t == EOT: break
            out.append(t); ids = ids + [t]
    return T.decode(out).strip()
def check(cond, msg):
    global ok; ok += bool(cond); print(f"[{'pass' if cond else 'FAIL'}] {msg}", flush=True)
base, sft = tiny(), tiny("sft_3e-4_600.pt")
a0, a1 = answer(base, "What did the cat do?"), answer(sft, "What did the cat do?")
print(f"  before: {a0!r}\n  after:  {a1!r}")
check(not a0.startswith("The cat") and a1 == "The cat sat on the mat.", "before fine-tuning Tiny continues the text; after, it answers 'The cat sat on the mat.' and stops")
c = corpora(); pool = [int(i) for i in c["pool"]]
import numpy as np
st = np.random.default_rng(SEED + 7).integers(0, len(pool) - 129, size=20); PW = torch.tensor([pool[s:s + 129] for s in st])
with torch.no_grad(): l0, l1 = float(base.loss(PW)), float(sft.loss(PW))
print(f"  loss on 20 windows of Tiny's own pool: before {l0:.4f}, after {l1:.4f}")
check(l1 > l0 + 1.0, "full fine-tuning on 500 pairs made Tiny more than a nat worse at its own training text (forgetting)")
log = load_json("sft_response")["log"]; s200 = [m for m in log if m["step"] == 200][0]["exact"]
print(f"  step 200: facts taught 12 times {s200['fact x12']}, 4 times {s200['fact x4']}, once {s200['fact x1']}")
num = lambda s: int(s.split("/")[0])
check(num(s200["fact x12"]) > num(s200["fact x4"]) > num(s200["fact x1"]), "facts taught more often were learnt first")
r = answer(sft, "Tell me what the mouse did.")
print(f"  reworded: 'Tell me what the mouse did.' -> {r!r}")
check(r != "The mouse slept on the stone.", "the same fact asked in other words is not answered (the wording was learnt, not the question)")
key = None
for line in open(os.path.join(HERE, "..", "..", ".env")) if os.path.exists(os.path.join(HERE, "..", "..", ".env")) else []:
    if "=" in line and "KEY" in line.split("=")[0]: key = line.split("=", 1)[1].strip().strip('"')
hit = key and any(key in open(f, errors="ignore").read() for f in glob.glob(os.path.join(HERE, "**", "*"), recursive=True) if os.path.isfile(f) and not f.endswith(".pt"))
check(key is not None and not hit, "the API key was read from .env and its string appears in no file under 04-code/ch38")
print(f"\n{ok} of 5 checkpoints pass")
