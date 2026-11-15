"""Figures for Chapter 26, Reproducibility.  Every number is read from 04-code/ch26/outputs/ (run the chapter's scripts
first: 04-code/ch26/RUN.txt).  Run from the project root: python3 03-figures/src/ch26_figs.py"""
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code"))
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch26"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch26", "outputs")
J = lambda n: json.load(open(os.path.join(CH, n + ".json")))
p = lambda n: os.path.join(OUT, n)
A, F = J("analyse"), J("fp_order")
R = {json.load(open(f))["tag"]: json.load(open(f)) for f in glob.glob(os.path.join(CH, "seeds_i*.json"))}

# 01 one million numbers summed in 200 orders (the 200 sums recomputed exactly as fp_order.py drew them)
rng = np.random.default_rng(1865)
x = rng.normal(0, 1, 1_000_000).astype(np.float32)
sums = [float(np.cumsum(x[rng.permutation(x.size)], dtype=np.float32)[-1]) for _ in range(200)]
a = F["A"]
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.hist(sums, bins=30, color=BLUE, alpha=0.85)
for k, col, ls, lab in (("exact", NAVY, "-", "exact"), ("pairwise", TEAL, "--", "pairwise"), ("left_to_right", AMBER, ":", "left to right"),
                        ("largest_first", VERM, "-.", "largest first")):
    ax.axvline(a[k], color=col, ls=ls, lw=2, label=f"{lab} {a[k]:.4f}")
ax.set_xlabel("sum of the same one million 32-bit numbers"); ax.set_ylabel("orders out of 200")
ax.set_title(f"One sum, 200 orders: {a['random_orders']['distinct']} different answers")
ax.legend(loc="upper left", fontsize=9)
save(fig, p("fig26-01-sums.png"))

# 02 how a difference grows: |batch-loss gap| against the one-thread run
base = np.array(R["i1865_o1865_t1"]["losses"])
fig, ax = plt.subplots(figsize=(8.6, 4.8))
for t, col, ls, lab in (("i1865_o1865_t2", TEAL, "-", "two threads"), ("i1865_o1865_t4", AMBER, "--", "four threads"),
                        ("i1865_o1865_t1_nudge", PLUM, "-.", "one-ulp nudge to one number"), ("i1865_o7_t1", VERM, ":", "another reading order")):
    d = np.abs(np.array(R[t]["losses"]) - base)
    s = np.arange(1, len(d) + 1)
    m = d > 0
    w = np.convolve(d, np.ones(10) / 10, mode="same")
    ax.plot(s[w > 0], w[w > 0], color=col, ls=ls, lw=2, label=lab)
ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("gap in batch loss from the one-thread run\n(10-step average)")
ax.set_title("Four ways to part from the same run")
ax.legend(loc="lower right", fontsize=9)
save(fig, p("fig26-02-drift.png"))

# 03 seeds: 3 x 3 grid and five-seed spreads (Looking-Glass)
g = A["grid_glass"]; G = np.array(g["table"])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
im = ax1.imshow(G, cmap="Blues_r"); ax1.grid(False)
for i in range(3):
    for j in range(3):
        ax1.text(j, i, f"{G[i, j]:.3f}", ha="center", va="center", fontsize=10, color="white" if G[i, j] < G.mean() else NAVY)
ax1.set_xticks(range(3)); ax1.set_xticklabels(["1865", "7", "11"]); ax1.set_yticks(range(3)); ax1.set_yticklabels(["1865", "7", "11"])
ax1.set_xlabel("reading-order seed"); ax1.set_ylabel("starting seed"); ax1.set_title("Looking-Glass loss, 400 steps")
f5 = A["five_glass"]
for k, (key, col, mk, lab) in enumerate((("init_only", BLUE, "o", "starting seed only"), ("order_only", AMBER, "s", "order seed only"), ("both", VERM, "D", "both"))):
    v = f5[key]
    ax2.scatter([k] * len(v), v, color=col, marker=mk, s=50, zorder=3)
    ax2.text(k, max(v) + 0.02, f"sd {np.std(v, ddof=1):.3f}", ha="center", fontsize=9, color=col)
ax2.set_xticks(range(3)); ax2.set_xticklabels(["starting seed\nonly", "order seed\nonly", "both"])
ax2.set_ylabel("Looking-Glass loss"); ax2.set_title("Five seeds each", pad=14)
ax2.set_xlim(-0.6, 2.6); ax2.set_ylim(min(min(f5[k]) for k in ("init_only", "order_only", "both")) - 0.03, max(max(f5[k]) for k in ("init_only", "order_only", "both")) + 0.06)
save(fig, p("fig26-03-seeds.png"))

# 04 a learning-rate comparison over five seeds
L = A["lr_glass"]
fig, ax = plt.subplots(figsize=(7.8, 4.8))
for i, (u, v) in enumerate(zip(L["lr1e3"], L["lr14e3"])):
    ax.plot([0, 1], [u, v], "-", color=GREY, lw=1.2)
ax.scatter([0] * 5, L["lr1e3"], color=BLUE, s=60, zorder=3, label="lr 1e-3")
ax.scatter([1] * 5, L["lr14e3"], color=VERM, marker="s", s=60, zorder=3, label="lr 1.4e-3")
txt = "difference by seed\n" + "\n".join(f"seed {s:>4}: {v - u:+.3f}" for u, v, s in zip(L["lr1e3"], L["lr14e3"], (1865, 7, 11, 21, 42)))
ax.text(1.12, np.mean(L["lr14e3"]), txt, va="center", fontsize=9, family="monospace")
ax.set_xticks([0, 1]); ax.set_xticklabels(["lr 1e-3", "lr 1.4e-3"]); ax.set_xlim(-0.3, 1.75)
ax.set_ylabel("Looking-Glass loss, 400 steps")
ax.set_title(f"Same five seeds, two learning rates: mean difference {L['mean']:+.4f} (se {L['se']:.3f})")
save(fig, p("fig26-04-lr.png"))

# 05 where two models disagree, by token frequency
D = A["disagree"]
fig, ax = plt.subplots(figsize=(8.8, 4.8))
names = list(D)
bands = list(D[names[0]]["bands"])
cols = [BLUE, AMBER, TEAL, PLUM, VERM]; lss = ["-", "--", "-.", ":", "-"]; mks = ["o", "s", "^", "D", "v"]
for nm, c, ls, mk in zip(names, cols, lss, mks):
    ax.plot(range(len(bands)), [D[nm]["bands"][b]["kl"] for b in bands], ls=ls, marker=mk, color=c, lw=2, label=nm)
ax.set_yscale("log"); ax.set_xticks(range(len(bands))); ax.set_xticklabels(bands)
ax.set_xlabel("how often the true next token occurs in the Carroll pool"); ax.set_ylabel("KL between the two models' shares (nats)")
ax.set_title("Where two runs disagree, Looking-Glass")
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.02, 0.55))
save(fig, p("fig26-05-disagree.png"))
print("fig26-01 to fig26-05 written to", OUT)
