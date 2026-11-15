"""Lab 49: checks on the recorded outputs of export49.py, traffic49.py and dash49.py.  Trains nothing; seconds."""
import os, json
from core49 import sha256, ART, HERE
O = os.path.join(HERE, "outputs"); n = ok = 0
def check(text, cond):
    global n, ok
    n += 1; ok += bool(cond); print(f"check {n}, {text}: {'pass' if cond else 'FAIL'}")
X = json.load(open(os.path.join(O, "export49.json"))); D = json.load(open(os.path.join(O, "dash49.json")))
man = json.load(open(os.path.join(ART, "MANIFEST.json")))
check("the 32-bit artefact gives the checkpoint's logits exactly", X["fp32"]["max_logit_diff"] == 0.0)
check("the 8-bit artefact keeps every top token on the running sentence", X["int8"]["same_top_token"])
check("the 8-bit file is under 30 per cent of the 32-bit file", X["int8"]["bytes"] < 0.3 * X["fp32"]["bytes"])
check("every file matches its manifest hash", all(sha256(os.path.join(ART, f)) == v["sha256"] for f, v in man["files"].items()))
check("the .pt checkpoint holds a pickle program that imports code", X["pt"]["global_imports"] > 0)
check(f"{D['all']['requests']:,} requests were logged", D["all"]["requests"] == 10000)
check("the first request was the running sentence and succeeded", D["first_log"]["prompt_tokens"] == 6 and D["first_log"]["status"] == 200)
pre = [p for p in D["probes"] if "/fp32/" in p["version"]]
check("before the 8-bit swap every probe matched Monday's exactly", all(p["max_change"] == 0.0 for p in pre))
q = [p for p in D["probes"] if "/int8/capstone" in p["version"]]
check("after the 8-bit swap the probes moved, but no top token changed", q and all(0 < p["max_change"] < 1 and p["top_changed"] == 0 for p in q))
V = D["versions"]; bad = [v for k, v in V.items() if k.endswith("tiny2048")][0]; good = [v for k, v in V.items() if k.endswith("int8/capstone")][0]
check("breakage visible: with the wrong tokenizer, errors appear and the share of known words falls by over 20 points",
      bad["codes"].get("500", 0) > 0 and good["known_share"] - bad["known_share"] > 0.2)
check("the 400 and 413 checks rejected the malformed requests", D["all"]["codes"].get("400", 0) > 0 and D["all"]["codes"].get("413", 0) > 0)
print(f"{ok} of {n} checks pass")
json.dump(dict(checks=n, passed=ok), open(os.path.join(O, "lab49.json"), "w"), indent=1)
