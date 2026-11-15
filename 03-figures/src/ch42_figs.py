"""Figures for Chapter 42, Mixture of Experts in Miniature.  Numbers are read from 04-code/ch42/outputs/ (run 04-code/ch42/RUN.txt first).
Run from the project root: python3 03-figures/src/ch42_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
setup()
OUT = os.path.join(ROOT, "03-figures", "ch42"); os.makedirs(OUT, exist_ok=True)
O = os.path.join(ROOT, "04-code", "ch42", "outputs")
J = lambda n: json.load(open(os.path.join(O, n + ".json")))
p = lambda n: os.path.join(OUT, n)

# 01 held-out loss against step for the first eight runs
runs = [("dense", "dense, seed 1865", GREY, "-"), ("dense_s2", "dense, seed 2", GREY, "--"), ("dense_wide", "dense wide (4 x width)", SLATE, ":"),
        ("moe4k1", "4 experts top-1, seed 1865", BLUE, "-"), ("moe4k1_s2", "4 experts top-1, seed 2", BLUE, "--"),
        ("moe4k1_noaux", "4 experts top-1, no balance loss", AMBER, "-"), ("moe8k1", "8 experts top-1", PLUM, "-"), ("moe8k2_fine", "8 small experts top-2", TEAL, "-")]
fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.2))
for name, lab, col, ls in runs:
    R = J(name); st = [x["step"] for x in R["log"]]; L = [x["held"] for x in R["log"]]
    axs[0].plot(st[1:], L[1:], color=col, ls=ls, lw=1.6, label=lab)
    axs[1].plot(st[4:], L[4:], color=col, ls=ls, lw=1.6, marker="o", ms=3)
axs[0].set_xlabel("step"); axs[0].set_ylabel("held-out loss (nats), 60 Looking-Glass windows"); axs[0].legend(fontsize=7, loc="upper right")
axs[0].set_title("Eight six-layer models, 2,000 steps", fontsize=9.5)
axs[1].set_xlabel("step"); axs[1].set_title("The last 1,000 steps, enlarged", fontsize=9.5)
fig.suptitle("A mixture of experts beside its dense twins", fontsize=11, x=0.01, ha="left")
save(fig, p("fig42-01-curves.png"))

# 02 token purity by layer, and capacity losses
A = {n: J("analyse_" + n) for n in ("moe4k1", "moe4k1_noaux", "moe8k2_fine", "moe4k1_bias")}
fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.2))
ax = axs[0]
cols = {"moe4k1": BLUE, "moe4k1_noaux": AMBER, "moe8k2_fine": TEAL, "moe4k1_bias": VERM}
labs = {"moe4k1": "4 experts top-1", "moe4k1_noaux": "4 experts top-1, no balance loss", "moe8k2_fine": "8 small experts top-2", "moe4k1_bias": "4 experts top-1, balanced by bias"}
for n, R in A.items():
    ax.plot(range(6), [l["purity"] for l in R["layers"]], color=cols[n], marker="o", label=labs[n])
    ax.plot(range(6), [l["purity_shuffled"] for l in R["layers"]], color=cols[n], ls=":", lw=1)
ax.set_ylim(0, 1.03); ax.set_xlabel("layer"); ax.set_ylabel("token purity")
ax.set_title("Routing is mostly by token, less so with depth (dotted: labels shuffled)", fontsize=9); ax.legend(fontsize=7, loc="lower left")
ax = axs[1]
w = 0.2
for i, (n, R) in enumerate(A.items()):
    xs = [j + (i - 1.5) * w for j in range(3)]
    ax.bar(xs, [c["loss"] - R["held"] for c in R["capacity"]], width=w, color=cols[n], label=labs[n])
ax.text(2, 0.02, "no slot dropped: 0", ha="center", fontsize=8); ax.set_xticks(range(3)); ax.set_xticklabels(["capacity 1.0", "capacity 1.25", "capacity 2.0"]); ax.set_ylabel("rise in held-out loss (nats)")
ax.set_title("A capacity limit at evaluation", fontsize=9.5); ax.legend(fontsize=7)
fig.suptitle("What the trained routers do", fontsize=11, x=0.01, ha="left")
save(fig, p("fig42-02-routers.png"))
print("ok")
