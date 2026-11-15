"""Collects every number Chapter 48's sections 48.6 to 48.10 quote, from outputs/ only, into outputs/facts48.json.
Run after eval48.py on each saved checkpoint of the main run."""
import os, json, glob, math
HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(HERE, "outputs")
M = [json.loads(l) for l in open(os.path.join(O, "main.jsonl"))][1:]
tr = [r for r in M if "loss" in r]; ev = {r["step"]: r["eval"] for r in M if "eval" in r}
F = dict(last_step=max(r["step"] for r in tr), evals={s: dict(held=e["heldout"]["loss"], held_bpc=e["heldout"]["bpc"], glass=e["glass"]["loss"], glass_bpc=e["glass"]["bpc"]) for s, e in ev.items()})
F["batch_loss"] = {r["step"]: r["loss"] for r in tr if r["step"] % 100 == 0}
late = [r for r in tr if r["step"] > 100]; g = max(late, key=lambda r: r["gnorm"])
F["gnorm_max_after100"] = dict(step=g["step"], value=g["gnorm"]); F["gnorm_last"] = tr[-1]["gnorm"]
F["gnorm_over1"] = [r["step"] for r in tr if r["gnorm"] > 1.0]
jumps = [(tr[i]["step"], tr[i]["loss"] - tr[i - 1]["loss"]) for i in range(1, len(tr)) if tr[i]["step"] > 200]
j = max(jumps, key=lambda x: x[1]); F["largest_rise"] = dict(step=j[0], value=j[1])
F["stalls"] = [(r["step"], r["tps"]) for r in tr if r["tps"] < 200]
F["tps_median"] = sorted(r["tps"] for r in tr)[len(tr) // 2]
F["eval_ckpt"] = {}
for f in glob.glob(os.path.join(O, "eval_main_step*.json")):
    R = json.load(open(f)); F["eval_ckpt"][R["step"]] = R
json.dump(F, open(os.path.join(O, "facts48.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in F.items() if k != "eval_ckpt"}, indent=0)[:3000])
for s, R in sorted(F["eval_ckpt"].items()):
    print(s, {k: round(R["fp32"][k]["bpc"], 4) for k in R["fp32"]}, {k: round(R["int8"][k]["bpc"], 4) for k in R["int8"]}, f"{R['cache_maxdiff']:.2e}", [round(x["tps"]) for x in R["samples"]])
