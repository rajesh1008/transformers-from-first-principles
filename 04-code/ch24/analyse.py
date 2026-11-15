"""Chapter 24: every table in the chapter, from outputs/run_*.json and outputs/inits_*.json.  python3 analyse.py"""
import _common
import json, os, glob, math
import numpy as np
from _common import HERE, save_json
O = os.path.join(HERE, "outputs")
def J(name):
    p = os.path.join(O, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None
BIGRAM = J("baselines")["carroll"]["bigram_k0.01"]
res = {}
print("Part 1. learning-rate sweep, 600 steps, warmup 50 then cosine to 0.1 of the peak; three reading orders")
print("peak lr   order 1865  order 7   order 11  mean     spread(max-min)  Looking-Glass(1865)  largest nudge norm, order 1865 (step)")
sweep = {}
for lr in ("5e-4", "1e-3", "2e-3", "4e-3", "8e-3"):
    r = J(f"run_cos_{lr}_600")
    if not r: continue
    st = r["log"]["steps"]; f = r["final"]
    gn = [x[2] for x in st]; k = int(np.argmax(gn))
    v = [f[1]] + [J(f"run_cos_{lr}_600_o{o}")["final"][1] if J(f"run_cos_{lr}_600_o{o}") else float("nan") for o in (7, 11)]
    have = [x for x in v if x == x]
    sweep[lr] = dict(orders=v, mean=float(np.mean(have)), spread=float(max(have) - min(have)))
    cell = lambda x: f"{x:.4f}" if x == x else "  -   "
    print(f"{lr:7s}   {cell(v[0])}      {cell(v[1])}    {cell(v[2])}    {np.mean(have):.4f}   {max(have) - min(have):.4f}           {f[2]:.4f}               {gn[k]:.3g} ({k})")
res["sweep"] = sweep
print(f"bigram predictor (baselines.py): {BIGRAM:.4f}")

print("\nPart 2. WSD: one trunk at a constant peak, branches decayed linearly to zero over a quarter of the branch point")
wsd = {}
for lr in ("1e-3", "2e-3"):
    r = J(f"run_wsd_{lr}_2000")
    if not r: continue
    tr = {e[0]: e for e in r["evals"]}
    for b, br in sorted(r["branches"].items(), key=lambda kv: int(kv[0])):
        b = int(b); f = br["final"]
        t_at = tr.get(b, (b, float("nan")))[1]
        le = max(k for k in tr if k <= br["length"]); t_end = tr[le][1]
        wsd[f"{lr}@{br['length']}"] = f[1]
        print(f"lr {lr} branch at {b:5d} -> finished at {br['length']:5d} steps ({br['passes']:.2f} passes): Carroll {f[1]:.4f} "
              f"Looking-Glass {f[2]:.4f} sentence {f[3]:.4f} | trunk at {b}: {t_at:.4f}; trunk at {le}: {t_end:.4f}; "
              f"decay gained {t_at - f[1]:.4f}")
res["wsd"] = wsd
print("cosine runs made for one length (warmup 50, cosine to 0.1 of the peak unless stated):")
for tag in ("cos_1e-3_625", "cos_1e-3_625_floor0", "cos_1e-3_1250", "cos_2e-3_1250", "cos_1e-3_1250_wd0.1"):
    r = J("run_" + tag)
    if r:
        f = r["final"]; res[tag] = f[1]
        print(f"  {tag:22s} Carroll {f[1]:.4f}  Looking-Glass {f[2]:.4f}  sentence {f[3]:.4f}  ({r['seconds']:.0f} s)")

print("\nPart 3. watching the 1e-3 trunk: batch loss, nudge norm, step-to-table ratio")
r = J("run_wsd_1e-3_2000")
if r:
    st = r["log"]["steps"]; L = np.array([x[1] for x in st]); gn = np.array([x[2] for x in st])
    for a, b in ((0, 50), (50, 200), (200, 500), (500, 1000), (1000, 1500), (1500, 2000)):
        print(f"  steps {a:4d}-{b:4d}: batch loss mean {L[a:b].mean():.4f}  nudge norm median {np.median(gn[a:b]):.3f}  max {gn[a:b].max():.3f}")
    spikes = [i for i in range(50, len(L)) if L[i] > L[i-50:i].mean() + 3 * L[i-50:i].std()]
    print(f"  steps whose batch loss is more than 3 spreads above the previous 50: {len(spikes)} {spikes[:12]}")
    big = [i for i in range(50, len(gn)) if gn[i] > 3 * np.median(gn[i-50:i])]
    print(f"  steps whose nudge norm is more than 3 x the median of the previous 50: {len(big)} {big[:12]}")
    rat = r["log"]["ratio"]
    print("  step-to-table ratio at steps", ", ".join(f"{s}: {v:.2e}" for s, v in rat if s in (0, 50, 100, 500, 1000, 1500, 1950)))
    ev = r["evals"]
    print("  held-out (Carroll) every 100 steps:", " ".join(f"{e[0]}:{e[1]:.3f}" for e in ev[::2]))
    # gap between training batches and held-out, near the end of the trunk
    print(f"  last 100 trunk batches mean {L[-100:].mean():.4f} against held-out at {ev[-1][0]}: {ev[-1][1]:.4f}")
    br = r["branches"]["2000"]
    bl = np.array([x[1] for x in br["log"]["steps"]])
    print(f"  2,500-step branch: last 100 batches mean {bl[-100:].mean():.4f} against held-out {br['final'][1]:.4f} (gap {br['final'][1] - bl[-100:].mean():.4f})")
    res["spikes"] = len(spikes); res["norm_jumps"] = len(big)

print("\nPart 4. Chapter 19's starts, plain gradient descent at lr 0.1, 400 steps, Looking-Glass held-out, ONE thread")
c19 = json.load(open(os.path.join(HERE, "..", "ch19", "outputs", "train_inits.json")))
names = {"gpt2": "0.02 everywhere", "scaled": "0.02, W_o and W_2 / sqrt(24)", "fan_in": "1/sqrt(fan-in)"}
inits = {}
for s in ("gpt2", "scaled", "fan_in"):
    a, b = J(f"inits_{s}_1865"), J(f"inits_{s}_7")
    ch = c19[f"{s}x1@0.1"]["held_end"]
    line = f"  {names[s]:30s} Chapter 19 (8 threads) {ch:.4f} | windows 1865, one thread {a['held_end']:.4f}" if a else ""
    if b: line += f" | windows 7, one thread {b['held_end']:.4f}"
    print(line)
    inits[s] = [ch, a["held_end"] if a else None, b["held_end"] if b else None]
res["inits"] = inits
save_json("analyse", res)
