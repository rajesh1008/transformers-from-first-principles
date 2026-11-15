"""Lab 24: predict, five breakages, checkpoint.  Needs outputs/ from run_all.sh and run_extra.sh (the 2,500-step
branch of the 1e-3 trunk is outputs/ckpt/wsd_1e-3_2000_b2000.pt).  One thread, 32-bit.  python3 lab24.py"""
import _common
import os, json, math
import numpy as np, torch
torch.set_num_threads(1)
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from _common import HERE, SEED, sentence_ids, tok
from _pool import heldout_sets, corpora, POOL_BOOKS
O = os.path.join(HERE, "outputs")
J = lambda n: json.load(open(os.path.join(O, n + ".json")))
base = J("baselines")["carroll"]
G, HB = heldout_sets()
c = corpora()
cfg = TorchTiny.from_numpy(Tiny(), dtype=torch.float32).config
def load(name):
    t = torch.load(os.path.join(O, "ckpt", name + ".pt"))
    return TorchTiny({k: v.numpy() for k, v in t.items()}, cfg, dtype=torch.float32)
tt = load("wsd_1e-3_2000_b2000")
L = lambda W: float(tt.loss(torch.as_tensor(np.asarray(W))))
ok = []
def check(name, cond, detail):
    ok.append(bool(cond)); print(f"[{'pass' if cond else 'FAIL'}] {name}: {detail}")
print("Part 1. predict, then read")
print(f"  uniform loss ln 2048 = {math.log(2048):.4f} nats = {math.log(2048) / math.log(2):.4f} bits")
print(f"  bigram perplexity e^{base['bigram_k0.01']:.4f} = {math.exp(base['bigram_k0.01']):.1f}")
r = J("run_wsd_1e-3_2000")
print(f"  steps in one run that gives finished runs of 625, 1,250 and 2,500 steps: 2,000 + 125 + 250 + 500 = 2,875 (logged: "
      f"{len(r['log']['steps']) + sum(len(b['log']['steps']) for b in r['branches'].values())})")
with torch.no_grad():
    print("\nPart 2. breakages")
    h = L(HB)
    print(f"  Breakage 1 (bits read as nats): the 2,500-step model's held-out loss {h:.4f} nats is {h / math.log(2):.4f} bits")
    pool = np.concatenate([np.asarray(c["books"][b]) for b in POOL_BOOKS])
    rng = np.random.default_rng(SEED)
    tr = np.stack([pool[a:a + 129] for a in rng.integers(0, len(pool) - 129, 16)])
    print(f"  Breakage 2 (scoring on training text): 16 windows of the pool {L(tr):.4f} against the held-out set {h:.4f}")
    held = np.asarray(c["held_b"])
    draws = []
    for s in range(20):
        g = np.random.default_rng(100 + s)
        draws.append(L(np.stack([held[a:a + 129] for a in g.integers(0, len(held) - 129, 16)])))
    allw = np.stack([held[a:a + 129] for a in range(0, len(held) - 129, 128)])
    print(f"  Breakage 3 (a small held-out set): 20 draws of 16 windows: {min(draws):.4f} to {max(draws):.4f} (spread {np.std(draws):.4f});"
          f" all {len(allw)} non-overlapping windows {L(allw):.4f}")
    tw = {e[0]: e[1] for e in r["evals"]}
    print(f"  Breakage 4 (the trunk read as a finished run): trunk at 1,000 {tw[1000]:.4f}, at 1,250 {tw[1200]:.4f} (step 1,200); "
          f"the branch finished at 1,250: {r['branches']['1000']['final'][1]:.4f}")
    c0, c1 = J("run_cos_1e-3_625")["final"][1], J("run_cos_1e-3_625_floor0")["final"][1]
    print(f"  Breakage 5 (a schedule that stops at a tenth of the peak): cosine to 0.1 {c0:.4f}, cosine to 0 {c1:.4f}, "
          f"WSD branch {r['branches']['500']['final'][1]:.4f} (625 steps each)")
    print("\nPart 3. checkpoints")
    check("the trained Tiny has Chapter 14's shape", sum(v.numel() for v in tt.t.values()) == 2766592,
          f"{sum(v.numel() for v in tt.t.values()):,} learned numbers")
    check("32-bit training repeats exactly at one thread (the 1e-3 sweep's first 600 steps against the trunk's)",
          all(a[1] == b[1] for a, b in zip(J("run_cos_1e-3_600")["log"]["steps"][:50], r["log"]["steps"][:50])),
          "first 50 batch losses identical (same warmup, same reader)")
    check("the 2,500-step model beats the bigram on the Carroll held-out set", h < base["bigram_k0.01"], f"{h:.4f} < {base['bigram_k0.01']:.4f}")
    check("the 2,500-step model is below the untrained Tiny on the running sentence", L(sentence_ids()) < r["evals"][0][3],
          f"{L(sentence_ids()):.4f} < {r['evals'][0][3]:.4f}")
    br = [r["branches"][k]["final"][1] for k in ("500", "1000", "2000")]
    check("each longer WSD branch ends lower", br[0] > br[1] > br[2], ", ".join(f"{x:.4f}" for x in br))
    T = tok(); ids = sentence_ids(T)
    print("\nwhat the 2,500-step model expects next, after each prefix of the running sentence (top 3, probabilities):")
    S = tt.scores(torch.tensor(ids))
    P = torch.softmax(S, -1)
    for i in range(len(ids) - 1):
        top = torch.topk(P[i], 3)
        pre = T.decode(ids[:i + 1])
        true_p = float(P[i, ids[i + 1]])
        print(f"  {pre!r:28s} -> " + ", ".join(f"{T.decode([int(j)])!r} {float(v):.3f}" for v, j in zip(top.values, top.indices))
              + f"   true {T.decode([ids[i + 1]])!r}: {true_p:.4f}")
print(f"\n{sum(ok)} of {len(ok)} checkpoints pass")
assert all(ok)
