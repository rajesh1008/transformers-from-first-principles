"""Figures for Chapter 31, Serving.  Numbers are read from 04-code/ch31/outputs/ (run 04-code/ch31/RUN.txt first).
Run from the project root: python3 03-figures/src/ch31_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
setup()
OUT = os.path.join(ROOT, "03-figures", "ch31"); os.makedirs(OUT, exist_ok=True)
J = lambda n: json.load(open(os.path.join(ROOT, "04-code", "ch31", "outputs", n + ".json")))
p = lambda n: os.path.join(OUT, n)
SC, EX = J("step_cost"), J("experiments")

# 01 decode tokens per second against batch size, two contexts
fig, ax = plt.subplots(figsize=(7.6, 5.6))
for ctx, col, mk, ls in ((128, BLUE, "o", "-"), (512, VERM, "s", "--")):
    rows = [r for r in SC["B"] if r["ctx"] == ctx]
    ax.plot([r["B"] for r in rows], [r["B"] / r["step"] for r in rows], marker=mk, ls=ls, color=col)
    ax.annotate(f"context {ctx}", ([r["B"] for r in rows][-1], [r["B"] / r["step"] for r in rows][-1]), xytext=(-70, 10),
                textcoords="offset points", color=col, fontsize=10)
ax.set_xscale("log", base=2); ax.set_xlabel("conversations per decode step (B)"); ax.set_ylabel("new tokens per second")
ax.set_title("GPT-2 small, one CPU thread: batching decode steps", fontsize=12)
save(fig, p("fig31-01-batch.png"))

# 02 throughput against p99 reply latency as the arrival rate grows
fig, ax = plt.subplots(figsize=(7.6, 5.6))
for sch, col, mk, ls in (("static", VERM, "^", ":"), ("continuous", BLUE, "o", "-"), ("chunked", TEAL, "s", "--")):
    rows = [r for r in EX["A"] if r["scheme"] == sch]
    ax.plot([r["tok_s"] for r in rows], [r["lat_p99"] for r in rows], marker=mk, ls=ls, color=col)
    ax.annotate(sch, (rows[-1]["tok_s"], rows[-1]["lat_p99"]), xytext=(4, 6), textcoords="offset points", color=col, fontsize=10)
ax.set_yscale("log"); ax.set_xlabel("output tokens per second (simulated on measured step costs)")
ax.set_ylabel("p99 reply latency, s (log scale)"); ax.set_title("Arrival rate 0.25 to 3 requests/s, 200 requests, B up to 16", fontsize=12)
save(fig, p("fig31-02-curve.png"))

# 03 chunk size against the longest gap and the median first-token time
ch = [32, 64, 128, 256, 512]
fig, ax = plt.subplots(figsize=(7.6, 5.6))
ax.plot(ch, [EX[f"C_{c}"]["gap_max"] * 1000 for c in ch], "o-", color=VERM)
ax.annotate("longest gap between tokens (ms)", (ch[2], EX["C_128"]["gap_max"] * 1000), xytext=(-60, 22), textcoords="offset points", color=VERM, fontsize=10)
ax.plot(ch, [EX[f"C_{c}"]["ttft_p50"] * 1000 for c in ch], "s--", color=BLUE)
ax.annotate("median time to first token (ms)", (ch[1], EX["C_64"]["ttft_p50"] * 1000), xytext=(10, 0), textcoords="offset points", color=BLUE, fontsize=10)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("prefill chunk, tokens per step"); ax.set_ylabel("milliseconds (log scale)")
ax.set_title("Chunked prefill at 1 request/s: one setting, two costs", fontsize=12)
save(fig, p("fig31-03-chunk.png"))
