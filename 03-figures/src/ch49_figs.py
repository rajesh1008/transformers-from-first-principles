"""Figures for Chapter 49 (partial, session 37).  fig49-01, the production dashboard after the simulated week, from
04-code/ch49/outputs/dash49.json (dash49.py).  Run from the project root: python3 03-figures/src/ch49_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt, numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch49"); os.makedirs(OUT, exist_ok=True)
D = json.load(open(os.path.join(ROOT, "04-code", "ch49", "outputs", "dash49.json")))
H = {int(k): v for k, v in D["hours"].items()}; hs = sorted(H); x = np.array(hs) / 24
ev = [(e["sim"] / 86400, ", ".join(f"{k} {v}" for k, v in e["change"].items())) for e in D["events"]]
fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True, gridspec_kw=dict(hspace=0.35))
a = axs[0]; a.bar(x, [H[h]["requests"] for h in hs], width=1 / 24, color=BLUE, alpha=0.85); a.set_title("Requests per hour")
b = axs[1]; b.plot(x, [H[h]["total_p50"] for h in hs], color=TEAL, lw=1.4, label="median"); b.plot(x, [H[h]["total_p95"] for h in hs], color=AMBER, lw=1.4, label="95th percentile")
b.set_title("Request latency, ms (successful requests)"); b.legend(frameon=False, fontsize=9)
c = axs[2]; c.plot(x, [100 * (1 - H[h]["codes"].get("200", 0) / H[h]["requests"]) for h in hs], color=VERM, lw=1.4, label="errors, % of requests")
c.plot(x, [100 * (H[h]["known_share"] or 0) for h in hs], color=PLUM, lw=1.2, label="output words found in the training books, %")
c.set_title("Errors and output quality per hour"); c.legend(frameon=False, fontsize=9); c.set_ylim(-3, 103)
P = D["probes"]; px = [p["sim"] / 86400 for p in P]; py = [p["max_change"] if p["max_change"] != "new token" else 10.0 for p in P]
d = axs[3]; d.step(px, [max(v, 1e-6) for v in py], where="post", color=NAVY, lw=1.6); d.set_yscale("log")
d.set_title("Probe drift: largest change in a probe log-probability since Monday 00:00 (10 = top tokens replaced)"); d.set_xlabel("day of the simulated week")
d.set_xticks(range(8)); d.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", ""])
for ax in axs:
    for t, lab in ev: ax.axvline(t, color=CORAL, ls="--", lw=1)
for t, lab in ev: axs[0].text(t + 0.02, axs[0].get_ylim()[1] * 0.95, lab, color=CORAL, fontsize=8, va="top", rotation=90)
fig.suptitle(f"The production dashboard: {D['all']['requests']:,} requests to the capstone over a simulated week", fontsize=14, color=NAVY, weight="bold")
save(fig, os.path.join(OUT, "fig49-01-dashboard.png")); print("ok")

# fig49-02 (session 38): latency against load, one point per simulated hour, and the latency distribution by stream mode.
import json as _j
R = [_j.loads(l) for l in open(os.path.join(ROOT, "04-code", "ch49", "logs", "requests.jsonl"))]
ok = [r for r in R if r["status"] == 200]
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(wspace=0.28))
rq = np.array([H[h]["requests"] for h in hs]); a.scatter(rq, [H[h]["total_p50"] for h in hs], s=14, color=TEAL, label="median")
a.scatter(rq, [H[h]["total_p95"] for h in hs], s=14, color=AMBER, label="95th percentile")
a.set_xlabel("requests in the simulated hour"); a.set_ylabel("total time, ms"); a.set_title("Latency does not grow with load at these rates"); a.legend(frameon=False, fontsize=9)
bins = np.linspace(0, 300, 61)
b.hist([r["total_ms"] for r in ok if not r["stream"]], bins=bins, color=BLUE, alpha=0.7, label="not streamed")
b.hist([r["total_ms"] for r in ok if r["stream"]], bins=bins, color=PLUM, alpha=0.7, label="streamed")
b.axvline(np.median([r["ttft_ms"] for r in ok]), color=CORAL, ls="--", lw=1.2, label="median time to first token")
b.set_xlabel("total time, ms (successful requests; axis cut at 300)"); b.set_ylabel("requests"); b.set_title("Where the time goes per request"); b.legend(frameon=False, fontsize=9)
fig.suptitle(f"Latency over {len(R):,} requests, one request at a time on one core", fontsize=13, color=NAVY, weight="bold")
save(fig, os.path.join(OUT, "fig49-02-latency.png")); print("ok 2")
