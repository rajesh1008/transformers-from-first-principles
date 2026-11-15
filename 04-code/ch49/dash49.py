"""Chapter 49, 49.4 and 49.6: the production dashboard from the logs.  Reads logs/requests.jsonl (serve49.py), outputs/probes49.jsonl
and outputs/traffic49.json (traffic49.py); writes outputs/dash49.json.  Per simulated day and per hour: requests, status codes,
latency (time to first token and total) at the 50th and 95th percentiles, tokens out, the share of output words that occur in the
capstone's training books (Chapter 48's words48.py measure), and the probe drift: the largest change in any of the 40 probe
log-probabilities from the first probe of the week."""
import os, re, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DAY = 86400
R = [json.loads(l) for l in open(os.path.join(HERE, "logs", "requests.jsonl"))]
P = [json.loads(l) for l in open(os.path.join(HERE, "outputs", "probes49.jsonl"))]
T = json.load(open(os.path.join(HERE, "outputs", "traffic49.json")))
W = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")
known = {w.lower() for w in W.findall(open(os.path.join(HERE, "..", "ch48", "corpus", "train.txt"), encoding="utf-8").read())}
pct = lambda a, q: float(np.percentile(a, q)) if len(a) else None
def summary(rs):
    ok = [r for r in rs if r["status"] == 200]; words = [w for r in ok for w in W.findall(r.get("text", ""))]
    return dict(requests=len(rs), codes={str(c): sum(r["status"] == c for r in rs) for c in sorted({r["status"] for r in rs})},
                ttft_p50=pct([r["ttft_ms"] for r in ok], 50), ttft_p95=pct([r["ttft_ms"] for r in ok], 95),
                total_p50=pct([r["total_ms"] for r in ok], 50), total_p95=pct([r["total_ms"] for r in ok], 95),
                tokens_in=sum(r.get("prompt_tokens", 0) for r in rs), tokens_out=sum(r.get("completion_tokens", 0) for r in ok),
                known_share=(sum(w.lower() in known for w in words) / len(words)) if words else None,
                ms_per_token=(sum(r["total_ms"] for r in ok) / max(1, sum(r["completion_tokens"] for r in ok))))
D = dict(all=summary(R), days={}, hours={}, versions={})
names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
for d in range(7): D["days"][names[d]] = summary([r for r in R if d * DAY <= int(r["sim"]) < (d + 1) * DAY])
for h in range(7 * 24):
    rs = [r for r in R if h * 3600 <= int(r["sim"]) < (h + 1) * 3600]
    if rs: D["hours"][h] = summary(rs)
for v in sorted({r["version"] for r in R}): D["versions"][v] = summary([r for r in R if r["version"] == v])
base = {p["prompt"]: dict(p["top"]) for p in P[0]["probes"]}
def drift(rec):
    m = 0.0; changed_top = 0
    for p in rec["probes"]:
        b = base[p["prompt"]]; top = dict(p["top"])
        changed_top += p["top"][0][0] != next(iter(b))
        for tid, lp in top.items(): m = max(m, abs(lp - b[tid]) if tid in b else float("inf"))
    return m, changed_top
D["probes"] = [dict(sim=p["sim"], version=p["version"], max_change=drift(p)[0] if drift(p)[0] != float("inf") else "new token", top_changed=drift(p)[1]) for p in P]
first = T["first"]; D["first"] = dict(request=first["request"], status=first["status"], answer=json.loads(first["answer"]) if first["answer"].startswith("{") else first["answer"])
D["first_log"] = R[0]
errs = {}
for r in R:
    if r["status"] != 200: errs[r.get("error", "")[:40]] = errs.get(r.get("error", "")[:40], 0) + 1
D["errors"] = errs; D["events"] = T["events"]; D["wall_seconds"] = T["wall_seconds"]
D["stream_broken"] = sum(1 for r in R if r["status"] == 500 and r.get("stream"))
D["client_codes"] = T["codes"]
json.dump(D, open(os.path.join(HERE, "outputs", "dash49.json"), "w"), indent=1)
print("all", D["all"]); [print(k, v) for k, v in D["days"].items()]; [print(k, v) for k, v in D["versions"].items()]
print("errors", errs); print("first", D["first_log"])
for p in D["probes"]:
    if p["sim"] % (12 * 3600) == 0 or p["top_changed"]: print("probe", p)
