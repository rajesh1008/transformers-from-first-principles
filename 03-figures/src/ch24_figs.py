"""Figures for Chapter 24, Training Tiny.  Every number is read from 04-code/ch24/outputs/*.json; run the chapter's
scripts first (04-code/ch24/RUN.txt).  Run from the project root: python3 03-figures/src/ch24_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch24"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch24", "outputs")
J = lambda n: json.load(open(os.path.join(CH, n + ".json")))
p = lambda n: os.path.join(OUT, n)
B = J("baselines")["carroll"]
BIG = B["bigram_k0.01"]

# 01 baselines
fig, ax = plt.subplots(figsize=(8.6, 3.8))
names = ["uniform", "untrained Tiny", "unigram", "bigram (+0.1)", "bigram (+0.01)"]
vals = [B["uniform"], B["untrained"], B["unigram"], B["bigram_k0.1"], B["bigram_k0.01"]]
ax.barh(names[::-1], vals[::-1], color=[BLUE, BLUE, AMBER, VERM, VERM][::-1])
for i, v in enumerate(vals[::-1]):
    ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)
ax.axvline(4.8904, ls="--", color=SLATE); ax.text(4.93, 4.3, "Chapter 23's best\none-pass model, 4.89", fontsize=8, color=SLATE)
ax.set_xlim(0, 8.6); ax.set_xlabel("held-out loss, Carroll held-out set (nats per token)")
ax.set_title("What a predictor gets without training Tiny")
save(fig, p("fig24-01-baselines.png"))

# 02 sweep
fig, ax = plt.subplots(figsize=(7.6, 4.2))
lrs = ["5e-4", "1e-3", "2e-3", "4e-3", "8e-3"]
ys = [J(f"run_cos_{l}_600")["final"][1] for l in lrs]
ax.plot([float(l) for l in lrs], ys, "o-", color=BLUE, label="order 1865")
o7 = [(float(l), J(f"run_cos_{l}_600_o7")["final"][1]) for l in lrs if os.path.exists(os.path.join(CH, f"run_cos_{l}_600_o7.json"))]
if o7:
    ax.plot(*zip(*o7), "s", color=AMBER, ms=8, label="order 7")
o11 = [(float(l), J(f"run_cos_{l}_600_o11")["final"][1]) for l in lrs if os.path.exists(os.path.join(CH, f"run_cos_{l}_600_o11.json"))]
if o11:
    ax.plot(*zip(*o11), "D", color=TEAL, ms=7, label="order 11")
ax.axhline(BIG, ls="--", color=VERM); ax.text(5.5e-4, BIG + 0.03, f"bigram {BIG:.2f}", color=VERM, fontsize=9)
ax.set_xscale("log"); ax.set_xlabel("peak learning rate"); ax.set_ylabel("held-out loss after 600 steps")
ax.legend(); ax.set_title("Tiny's own learning-rate sweep (600 steps, cosine)")
save(fig, p("fig24-02-sweep.png"))

# 03 WSD trunk and branches
fig, ax = plt.subplots(figsize=(9.0, 4.6))
for lr, col in (("1e-3", BLUE), ("2e-3", AMBER)):
    r = J(f"run_wsd_{lr}_2000")
    ev = np.array(r["evals"]); ax.plot(ev[1:, 0], ev[1:, 1], "-", color=col, label=f"trunk, constant {lr}")
    for k, b in r["branches"].items():
        be = np.array(b["evals"]); ax.plot(be[:, 0], be[:, 1], "-", color=col, lw=2.4)
        ax.plot(be[-1, 0], be[-1, 1], "o", color=col)
for tag, mk in (("cos_1e-3_625", "^"), ("cos_1e-3_1250", "^"), ("cos_2e-3_1250", "v")):
    if os.path.exists(os.path.join(CH, f"run_{tag}.json")):
        r = J(f"run_{tag}"); ax.plot(r["final"][0], r["final"][1], mk, color=VERM, ms=9)
ax.plot([], [], "^", color=VERM, label="cosine runs made for one length")
ax.axhline(BIG, ls="--", color=SLATE); ax.text(120, BIG - 0.12, f"bigram {BIG:.2f}", color=SLATE, fontsize=9)
ax.set_xscale("log"); ax.set_xlabel("step"); ax.set_ylabel("held-out loss (Carroll)")
ax.legend(); ax.set_title("One WSD trunk, three decayed branches")
save(fig, p("fig24-03-wsd.png"))

# 04 instruments of the 1e-3 trunk
r = J("run_wsd_1e-3_2000")
st = np.array(r["log"]["steps"]); ra = np.array(r["log"]["ratio"])
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9.0, 7.4), sharex=True)
a1.plot(st[:, 0], st[:, 1], color=BLUE, lw=0.6); a1.set_ylabel("batch loss")
ev = np.array(r["evals"]); a1.plot(ev[:, 0], ev[:, 1], "o-", color=VERM, ms=3, label="held-out every 100 steps"); a1.legend()
a2.plot(st[:, 0], st[:, 2], color=AMBER, lw=0.6); a2.set_yscale("log"); a2.set_ylabel("nudge norm")
a3.plot(ra[:, 0], ra[:, 1], "o-", color=NAVY, ms=3); a3.set_yscale("log"); a3.set_ylabel("step / tables")
a3.set_xlabel("step"); a1.set_title("The 1e-3 trunk's instruments, 2,000 steps")
save(fig, p("fig24-04-instruments.png"))

# 05 Chapter 19 starts, two window seeds
A = J("analyse")["inits"]
fig, ax = plt.subplots(figsize=(8.0, 4.0))
lab = {"gpt2": "0.02 everywhere", "scaled": "0.02, writers / sqrt(24)", "fan_in": "1/sqrt(fan-in)"}
x = np.arange(3)
for j, (nm, col) in enumerate((("Chapter 19, 8 threads", SLATE), ("windows 1865, one thread", BLUE), ("windows 7, one thread", AMBER))):
    v = [A[s][j] if A[s][j] is not None else np.nan for s in lab]
    ax.bar(x + (j - 1) * 0.27, v, 0.27, color=col, label=nm)
ax.set_xticks(x); ax.set_xticklabels(list(lab.values())); ax.set_ylim(5.0, 6.2)
ax.set_ylabel("held-out loss, Looking-Glass"); ax.legend(fontsize=8); ax.set_title("Chapter 19's starts, second seed (SGD 0.1, 400 steps)")
save(fig, p("fig24-05-seeds.png"))
print("figures written to", OUT)
