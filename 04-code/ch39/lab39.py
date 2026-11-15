"""Lab 39: build a low-rank adapter by hand, check that it starts as a no-op, count what it trains, merge it, and read
the comparison runs.  Needs tune.py's outputs (tune_lora_r8all_1800.json and its saved A and B, tune_lora_r36.json,
tune_lora_r8all.json, tune_lora_r8all_initb.json; see RUN.txt).  Six checkpoints; CPU, one thread, 32-bit, seed 1865.
Makes no API call; reads .env only to check that the key string appears in no file under 04-code/ch39."""
from _common import *
import glob, torch, numpy as np
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
from _pool import corpora
ok = 0
def check(cond, msg):
    global ok; ok += bool(cond); print(f"[{'pass' if cond else 'FAIL'}] {msg}", flush=True)
base = os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")
tt = load(base); W0 = {k: v.clone() for k, v in tt.t.items()}
pool = [int(i) for i in corpora()["pool"]]
st = np.random.default_rng(SEED + 7).integers(0, len(pool) - 129, size=20); PW = torch.tensor([pool[s:s + 129] for s in st])
r = 8; names = [k for k in tt.t if k.startswith("h.") and k.split(".")[-1].startswith("W_")]
AB = {k: (torch.randn(W0[k].shape[0], r) / W0[k].shape[0] ** 0.5, torch.zeros(r, W0[k].shape[1])) for k in names}
with torch.no_grad():
    s0 = tt.scores(PW[:, :-1])
    for k, (A, B) in AB.items(): tt.t[k] = W0[k] + 2.0 * (A @ B)
    s1 = tt.scores(PW[:, :-1])
d = float((s0 - s1).abs().max()); print(f"  adapter with B = 0 on {len(names)} matrices: largest change in any score {d:.1e}")
check(d == 0.0, "an adapter whose B starts at zero leaves every score exactly as it was")
n = sum(A.numel() + B.numel() for A, B in AB.values()); tot = sum(v.numel() for v in W0.values())
print(f"  rank 8 on every block matrix: {n:,} trainable numbers of {tot:,} ({100 * n / tot:.2f} per cent)")
check(n == 221184, "rank 8 on all 72 block matrices trains 221,184 numbers, 8 per cent of Tiny")
m = load_json("merge"); print(f"  merged against separate: largest score difference {m['maxdiff']:.1e}; pool loss {m['loss_sep']:.4f} / {m['loss_merged']:.4f}")
check(m["maxdiff"] < 1e-4 and m["ranks"] == [8], "merging the adapter into the tables gives the same scores, and every learned change has rank 8")
last = lambda name: load_json(name)["log"][-1]
num = lambda s: int(s.split("/")[0])
a, b = last("tune_lora_r36"), last("tune_lora_r8all")
print(f"  221,184 numbers each, 600 steps: attention only (rank 36) facts {a['exact']['fact']}, pool {a['pool']:.4f}; every matrix (rank 8) facts {b['exact']['fact']}, pool {b['pool']:.4f}")
check(num(b["exact"]["fact"]) > num(a["exact"]["fact"]), "with the same number of trainable numbers, adapters on every matrix learnt more facts than adapters on the attention maps")
ib = load_json("tune_lora_r8all_initb")["log"][0]
print(f"  B started random instead of zero: step-0 pool loss {ib['pool']:.4f} against 3.5367")
check(ib["pool"] > 3.5367 + 1.0, "an adapter that does not start at zero damages the model before the first step (breakage)")
key = None
envp = os.path.join(HERE, "..", "..", ".env")
for line in open(envp) if os.path.exists(envp) else []:
    if "=" in line and "KEY" in line.split("=")[0]: key = line.split("=", 1)[1].strip().strip('"')
hit = key and any(key in open(f, errors="ignore").read() for f in glob.glob(os.path.join(HERE, "**", "*"), recursive=True) if os.path.isfile(f) and not f.endswith(".pt"))
check(key is not None and not hit, "the API key was read from .env and its string appears in no file under 04-code/ch39")
print(f"\n{ok} of 6 checkpoints pass")
