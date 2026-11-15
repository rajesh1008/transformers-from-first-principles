"""Figures for Chapter 28, Decoding Strategies in Depth.  Every number is read from 04-code/ch28/outputs/ (run the chapter's
scripts first: 04-code/ch28/RUN.txt).  Run from the project root: python3 03-figures/src/ch28_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch28"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch28", "outputs")
J = lambda n: json.load(open(os.path.join(CH, n + ".json")))
p = lambda n: os.path.join(OUT, n)
C, F, X, N = J("contrastive"), J("fair"), J("exact_constrained"), J("best_of_n")

# 01 contrastive decoding, greedy rows
rows = [k for k in C if k.startswith("A ")]
labs = [k[2:].replace(" b1.0", "").replace("greedy", "").strip() or k for k in rows]
x = np.arange(len(rows)); w = 0.38
fig, ax = plt.subplots(figsize=(9.6, 4.8))
ax.bar(x - w / 2, [C[k]["rep4"] for k in rows], w, color=BLUE, label="repeated 4-token runs")
ax.bar(x + w / 2, [C[k]["novel"] for k in rows], w, color=AMBER, label="words not in the pool")
ax.set_xticks(x); ax.set_xticklabels([k[2:] for k in rows], rotation=30, ha="right", fontsize=8.5)
ax.set_ylabel("share"); ax.legend(fontsize=9)
ax.set_title("Greedy decoding, 60 tokens x 3 prompts: the expert, two amateurs, and their contrast")
save(fig, p("fig28-01-contrastive.png"))

# 02 curves, not points
R = ["temperature", "top-p", "min-p", "typical", "mirostat"]; pts = F["points"]
fig, ax = plt.subplots(figsize=(8.8, 5.0))
for r, col in zip(R, SERIES):
    for sd, ls, a in [("1865", "-", 1.0), ("7", ":", 0.55), ("11", "--", 0.55)]:
        v = sorted((pts[k]["surprise"], pts[k]["novel"]) for k in pts if k.split("|")[0] == r and k.split("|")[2] == sd)
        ax.plot([q[0] for q in v], [q[1] for q in v], ls, marker="o", ms=4, color=col, alpha=a, label=r if sd == "1865" else None)
ax.set_xlabel("mean surprise of the chosen tokens (nats, untempered model)"); ax.set_ylabel("share of words not in the pool")
ax.set_title("Five rules x five settings x three seed lists (solid 1865, dotted 7, dashed 11)")
ax.legend(fontsize=9)
save(fig, p("fig28-02-curves.png"))

# 03 exact against masked
names = ["Alice", "Bruno", "Sylvie", "Balbus", "the Hatter", "the Queen"]
x = np.arange(6); w = 0.38
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.bar(x - w / 2, X["names"]["exact"], w, color=BLUE, label="exact conditional (the model's own choice among six)")
ax.bar(x + w / 2, X["names"]["masked"], w, color=VERM, label="token-by-token masking")
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("chance"); ax.legend(fontsize=9)
ax.set_title("Six names: exact sampling against masking (total-variation distance 0.500)")
save(fig, p("fig28-03-exact.png"))

# 04 sampling many and choosing
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.4))
ns = [1, 4, 16]
a1.plot(ns, [N[f"A n{n}"]["surprise"] for n in ns], "o-", color=BLUE, lw=2, label="mean surprise (nats)")
a1.set_xscale("log", base=2); a1.set_xticks(ns); a1.set_xticklabels(ns); a1.set_xlabel("n samples, best kept by the model's own score")
a1.set_ylabel("mean surprise (nats)", color=BLUE)
b = a1.twinx(); b.plot(ns, [N[f"A n{n}"]["novel"] for n in ns], "s--", color=AMBER, lw=2); b.set_ylabel("words not in the pool", color=AMBER); b.grid(False)
a1.set_title("Best-of-n by likelihood")
ks = ["greedy", "one sample", "vote of 4", "vote of 16", "any of 4", "any of 16"]
a2.bar(range(6), [N["B " + k] for k in ks], color=[GREY, BLUE, BLUE, BLUE, GREEN, GREEN])
a2.set_xticks(range(6)); a2.set_xticklabels(ks, rotation=25, ha="right", fontsize=9); a2.set_ylabel("share of 150 held-out words right")
a2.set_title("Next whole word: one, voted, any")
save(fig, p("fig28-04-many.png"))
print("ok")
