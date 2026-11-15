"""Chapter 25, section 25.8: a planted contamination, measured, and the n-gram check that finds it (or does not).
Start from the finished 2,500-step Tiny (Chapter 24's tables; Adam's averages were not saved, so a fresh Adam is used)
and train 100 more steps on the Carroll pool (8 windows a step, ShardStream order 1865, Adam b2 0.95, no decay,
learning rate 1e-4 falling in a straight line to zero).  Four runs read the same batches, except that in three of them
one window of some batches is replaced by one of 8 "planted" windows of the Carroll held-out set (windows 0, 4, 8, ..., 28
of the 31):
  control       no planted windows
  exact x1      each planted window read once (8 batches)
  exact x4      each read four times (32 batches)
  altered x4    each read four times, with every eighth token replaced by a random pool token (seed 1865), so that
                no 13-token stretch of a planted copy matches the held-out text
Measured: loss on the 8 planted windows, on the 23 others, on the whole held-out set, and on 64 Looking-Glass windows.
Then the n-gram check: the share of 13-token (and 8-token) stretches of each held-out window that occur in the training
text (the pool plus whatever was planted), and the natural overlap of the held-out sets with the pool at n = 4, 8, 13.
32-bit, one thread.  Writes outputs/contaminate.json."""
import _common
import os, math, json
import numpy as np, torch
torch.set_num_threads(1); torch.manual_seed(1865)
from tiny import eval as E
from tiny.optim import Adam
from tiny import data as D
from _pool import corpora, shards, POOL_BOOKS
from _common import HERE, SEED, save_json
c = corpora()
pool = np.concatenate([np.asarray(c["books"][b]) for b in POOL_BOOKS])
HW = E.windows(c["held_b"])                          # 31 windows
PL = list(range(0, 31, 4))                            # 8 planted windows
OTHER = [i for i in range(31) if i not in PL]
GW = E.windows(c["glass"])[:64]
CK = os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")
rng = np.random.default_rng(SEED)
ALT = HW[PL].copy()
for w in ALT:
    w[::8] = pool[rng.integers(0, len(pool), size=len(w[::8]))]
STEPS = 100

def losses(tt):
    with torch.no_grad():
        f = lambda W: float(tt.loss(torch.tensor(W)))
        return dict(planted=f(HW[PL]), other=f(HW[OTHER]), all31=f(HW), glass64=f(GW))

def run(kind):
    tt = E.load(CK)
    for v in tt.t.values():
        v.requires_grad_(True)
    opt = Adam(1e-4)
    stream = D.ShardStream(shards(), window=129, batch=8, buffer=256, seed=SEED)
    copies = {"control": 0, "exact x1": 1, "exact x4": 4, "altered x4": 4}[kind]
    src = ALT if kind.startswith("altered") else HW[PL]
    plan = {}
    if copies:
        slots = np.linspace(0, STEPS - 1, 8 * copies).round().astype(int)      # spread over the run
        for j, s in enumerate(slots):
            plan.setdefault(int(s), []).append(src[j % 8])
    for s in range(STEPS):
        X = stream.next_batch()
        for k, w in enumerate(plan.get(s, [])):
            X[k] = w
        for v in tt.t.values():
            v.grad = None
        L = tt.loss(torch.tensor(X)); L.backward()
        opt.step(tt.t, (STEPS - s) / STEPS)
    return losses(tt)

out = dict(start=losses(E.load(CK)))
print("Part 1. planted contamination: 100 more steps from the 2,500-step model, lr 1e-4 to zero, same batches")
print(f"  {'run':12} {'planted 8':>9} {'other 23':>9} {'all 31':>8} {'LG 64':>8}")
r = out["start"]; print(f"  {'start':12} {r['planted']:9.4f} {r['other']:9.4f} {r['all31']:8.4f} {r['glass64']:8.4f}")
for kind in ("control", "exact x1", "exact x4", "altered x4"):
    r = out[kind] = run(kind)
    print(f"  {kind:12} {r['planted']:9.4f} {r['other']:9.4f} {r['all31']:8.4f} {r['glass64']:8.4f}", flush=True)
for kind in ("exact x1", "exact x4", "altered x4"):
    d = {k: out[kind][k] - out["control"][k] for k in ("planted", "other", "all31")}
    print(f"  {kind} against control: planted {d['planted']:+.4f}, other {d['other']:+.4f}, whole held-out set {d['all31']:+.4f}")

print("\nPart 2. the n-gram check (token n-grams of each held-out window found in the training text)")
for n in (13, 8):
    base = E.ngrams(pool, n)
    for kind, extra in (("pool only", []), ("pool + exact copies", list(HW[PL])), ("pool + altered copies", list(ALT))):
        S = set(base)
        for w in extra:
            S |= E.ngrams(w, n)
        sh = [E.overlap(S, w, n)[0] for w in HW]
        flag = [i for i, x in enumerate(sh) if x >= 0.5]
        print(f"  n = {n:2d}, {kind:22}: planted windows overlap {np.mean([sh[i] for i in PL]):.3f}, others {np.mean([sh[i] for i in OTHER]):.3f}; "
              f"windows with half or more of their {n}-grams found: {flag}")
        out[f"ngram_{n}_{kind}"] = dict(planted=float(np.mean([sh[i] for i in PL])), other=float(np.mean([sh[i] for i in OTHER])), flagged=flag)
print("  natural overlap with the pool (share of positions whose n-gram occurs in the pool):")
alice = np.asarray(c["alice"])
for n in (4, 8, 13):
    P = E.ngrams(pool, n); A = E.ngrams(alice, n)
    a = E.overlap(P, c["held_b"], n)[0]; g = E.overlap(P, c["glass"], n)[0]; ga = E.overlap(A, c["glass"], n)[0]
    print(f"    n = {n:2d}: Carroll held-out {a:.4f}  Looking-Glass {g:.4f}  (Looking-Glass against Alice: {ga:.4f})")
    out[f"natural_{n}"] = dict(carroll=a, glass=g, glass_alice=ga)
save_json("contaminate", out)
