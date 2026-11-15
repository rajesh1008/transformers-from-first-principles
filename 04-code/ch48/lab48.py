"""Lab 48: checks on the capstone's recorded outputs.  Reads outputs/ (written by tok48.py, train48.py and eval48.py) and the
model code; trains nothing.  Run after run_main.sh, run_sweep.sh and eval48.py ckpt/main_step2000.pt.  CPU, seconds."""
import os, sys, json, math, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from tiny.capstone.model import Capstone
O = os.path.join(HERE, "outputs"); n = 0; ok = 0
def rows(name): return [json.loads(l) for l in open(os.path.join(O, name + ".jsonl"))]
def last_eval(name): return [r for r in rows(name) if "eval" in r][-1]
def check(text, cond):
    global n, ok
    n += 1; ok += bool(cond); print(f"check {n}, {text}: {'pass' if cond else 'FAIL'}")
V, W, L = 8192, 256, 10
hand = V * W + 256 * W + L * (2 * 2 * W + 3 * W * W + W * W + (W * 4 * W + 4 * W) + (4 * W * W + W)) + 2 * W
check(f"the hand count equals the model's ({hand:,})", hand == Capstone().count() == 10_050_560)
T = json.load(open(os.path.join(O, "tok48.json")))
check("the running sentence is six tokens with the 8,192 vocabulary", len(T["sentence"]["ids"]) == 6)
M = rows("main"); start = M[1]["eval"]["heldout"]["loss"]
check(f"the start is near uniform: {start:.4f} within 0.1 of ln 8192 = {math.log(V):.4f}", abs(start - math.log(V)) < 0.1)
fin = [r for r in M if r.get("step") == 2000]
check("the main run reached step 2,000 with a finite loss", bool(fin) and math.isfinite(fin[0]["loss"]) and not any(r.get("diverged") for r in M))
ev = [r for r in M if "eval" in r]
check("held-out bits per character on The Time Machine fell from the start to the end", ev[-1]["eval"]["heldout"]["bpc"] < ev[0]["eval"]["heldout"]["bpc"])
E = sorted(glob.glob(os.path.join(O, "eval_main_step*.json")), key=lambda f: int(f.split("step")[-1].split(".")[0]))
R = json.load(open(E[-1])) if E else None
check("8-bit rounding changes held-out bits per character by less than 0.01", R and abs(R["int8"]["heldout"]["bpc"] - R["fp32"]["heldout"]["bpc"]) < 0.01)
check("the KV-cache loop matches the full forward pass within 1e-4", R and R["cache_maxdiff"] < 1e-4)
check("three seeded samples were written", R and len(R["samples"]) == 3)
done = all(last_eval(p + lr)["step"] == 400 for p, lrs in (("s10m_", ("3e-4", "1e-3", "3e-3", "1e-2")), ("s2m_", ("1e-3", "3e-3", "1e-2", "3e-2"))) for lr in lrs)
check("all eight sweep runs reached step 400", done)
s10 = {lr: last_eval("s10m_" + lr)["eval"]["heldout"]["loss"] for lr in ("3e-4", "1e-3", "3e-3", "1e-2")}
s2 = {lr: last_eval("s2m_" + lr)["eval"]["heldout"]["loss"] for lr in ("1e-3", "3e-3", "1e-2", "3e-2")}
b10, b2 = min(s10, key=s10.get), min(s2, key=s2.get)
print(f"  10M at 400 steps: " + ", ".join(f"{k} {v:.4f}" for k, v in s10.items()) + f"; best {b10}")
print(f"  1.9M at 400 steps: " + ", ".join(f"{k} {v:.4f}" for k, v in s2.items()) + f"; best {b2}")
check("the smaller model's best rate is higher than the larger model's", float(b2) > float(b10))
check(f"breakage visible: the 1.9M model's best rate ({b2}) costs the 10M model loss ({s10.get(b2, float('nan')):.4f} against {s10[b10]:.4f})", s10.get(b2, 0) > s10[b10])
print(f"{ok} of {n} checks pass")
json.dump(dict(checks=n, passed=ok, s10=s10, s2=s2, best10=b10, best2=b2, main_eval=E[-1] if E else None), open(os.path.join(O, "lab48.json"), "w"), indent=1)
