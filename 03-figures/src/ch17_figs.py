"""Figures for Chapter 17, Learning as Nudging.  Every number is read from 04-code/ch17/outputs or recomputed
with the hand Tiny of Chapter 15 (04-code/ch15/hand.py).  Run from the project root:
python3 03-figures/src/ch17_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(ROOT, "04-code"), os.path.join(ROOT, "04-code", "ch15")):
    sys.path.insert(0, p)
from style import *
import matplotlib.pyplot as plt
import numpy as np
from hand import hand_tiny, IDS
setup()
OUT = os.path.join(ROOT, "03-figures", "ch17"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch17", "outputs")
js = lambda n: json.load(open(os.path.join(CH, n + ".json")))
ONE, HILL, HN, TH, TT, COST, TC = (js(n) for n in ("one_number", "loss_hill", "head_nudges", "train_hand", "train_tiny", "cost", "tiny_check"))

m = hand_tiny(); S, st = m.forward(IDS, keep=True); TOP = m.top_rows(st[-1]); E0 = m.E.table.copy()
def scores(x):
    e = E0.copy(); e[4, 0] = x; return TOP[:5] @ e.T
def surprise_mat(x):
    z = scores(x)[4]; z = z - z.max(); return float(-(z[4] - np.log(np.exp(z).sum())))
def loss(x):
    z = scores(x); z = z - z.max(1, keepdims=True)
    return float(-(z - np.log(np.exp(z).sum(1, keepdims=True)))[np.arange(5), IDS[1:]].mean())

# 01 one learned number: the surprise on mat as x moves, the tangent at x = -1, three steps
xs = np.linspace(-4, 6, 400)
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.plot(xs, [surprise_mat(x) for x in xs], color=BLUE, label="surprise on 'mat' after the@4")
r0 = ONE["rows"][0]
tx = np.linspace(-3, 1, 10); ax.plot(tx, r0["surprise"] + r0["slope"] * (tx + 1), color=VERM, ls="--", lw=1.6, label="slope at x = -1: -1.1765")
for r in ONE["rows"]:
    ax.plot(r["x"], r["surprise"], "o", color=NAVY, ms=7)
    ax.annotate(f"step {r['step']}", (r["x"], r["surprise"]), textcoords="offset points", xytext=(6, 6), fontsize=9, color=NAVY)
ax.set_xlabel("x, mat's first learned number"); ax.set_ylabel("surprise"); ax.legend(loc="upper right")
ax.set_title("One number, nudged downhill three times (learning rate 1)")
save(fig, os.path.join(OUT, "fig17-01-one-number.png"))

# 02 the slope as a shrinking wiggle, on x squared at x = 1.5
f = lambda x: x ** 2
fig, ax = plt.subplots(figsize=(8.0, 4.4))
xx = np.linspace(-0.2, 3.2, 200); ax.plot(xx, f(xx), color=BLUE, label="x squared")
for h, col in ((1.5, AMBER), (0.5, TEAL), (0.05, VERM)):
    s = (f(1.5 + h) - f(1.5)) / h
    ax.plot(xx, f(1.5) + s * (xx - 1.5), color=col, lw=1.3, ls="--", label=f"wiggle {h}: change / wiggle = {s:.2f}")
ax.plot(1.5, f(1.5), "o", color=NAVY); ax.set_ylim(-1, 10); ax.legend(loc="upper left", fontsize=9)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("The slope is what change / wiggle settles to: 3.0 at x = 1.5")
save(fig, os.path.join(OUT, "fig17-02-slope.png"))

# 03 the chain rule: nudges multiply
fig, ax = canvas(9.0, 3.4)
box(ax, 3, 12, 16, 10, "x = 2", color=NAVY)
box(ax, 37, 12, 16, 10, "u = 3x + 1 = 7", color=BLUE, fs=10)
box(ax, 71, 12, 16, 10, "y = u² = 49", color=TEAL, fs=10)
arrow(ax, 19, 17, 37, 17); arrow(ax, 53, 17, 71, 17)
label(ax, 28, 25, "slope 3", color=BLUE, fs=11, weight="bold"); label(ax, 62, 25, "slope 2u = 14", color=TEAL, fs=11, weight="bold")
label(ax, 45, 5, "slope of y with respect to x = 3 × 14 = 42 (the wiggle gives 42.0000)", color=VERM, fs=12, weight="bold")
label(ax, 45, 31, "Nudges multiply along a chain", color=NAVY, fs=13, weight="bold")
save(fig, os.path.join(OUT, "fig17-03-chain-rule.png"))

# 04 the learning-rate ladder on the one-number loss
fig, axs = plt.subplots(1, 4, figsize=(12.4, 3.6), sharey=True)
g = np.array(ONE["grid"]); Lg = np.array(ONE["L"])
for ax, key, col in zip(axs, ("0.500", "2.000", "8.960", "20.000"), (TEAL, BLUE, AMBER, VERM)):
    run = ONE["runs"][key]; p = np.array(run["path"]); lp = [loss(x) for x in p]
    gw = np.linspace(-11.5, 6, 300); ax.plot(gw, [loss(x) for x in gw], color=GREY, lw=1.6)
    ax.plot(p, lp, "-o", color=col, ms=4, lw=1.0)
    ax.set_title(f"learning rate {float(key):g}", fontsize=11); ax.set_xlim(-11.5, 6)
    ax.set_xlabel("x")
axs[0].set_ylabel("loss (five seats)")
fig.suptitle("Too small crawls, about right arrives, too big bounces (12 steps from x = -1; bottom 2.9126 at x = 1.6303)", color=NAVY, fontsize=12, x=0.02, y=1.04, ha="left")
save(fig, os.path.join(OUT, "fig17-04-lr-ladder.png"))

# 05 Signature 6: the Loss Hill over mat's two numbers
X, Y = np.meshgrid(HILL["xs"], HILL["ys"]); G = np.array(HILL["G"])
fig, ax = plt.subplots(figsize=(8.6, 6.6))
cs = ax.contourf(X, Y, G, levels=24, cmap="Blues_r"); ax.contour(X, Y, G, levels=24, colors="white", linewidths=0.4)
fig.colorbar(cs, ax=ax, label="loss")
for key, col, n, z in (("14.0", VERM, 12, 3), ("6.5", AMBER, 121, 4), ("1.0", "#7fe0c8", 121, 5)):
    p = np.array(HILL["paths"][key]["path"])[:n]
    ax.plot(p[:, 0], p[:, 1], "-o" if n < 50 else "-", ms=3, color=col, lw=1.6, zorder=z,
            label=f"learning rate {float(key):g}" + (f", first {n} steps" if n < 50 else f", first {n - 1} steps"))
ax.set_xlim(-6, 4); ax.set_ylim(-2, 10)
ax.plot(-1, 1, "o", color="white", ms=8, zorder=6); ax.annotate("start (-1, 1), loss 3.3313", (-1, 1), xytext=(8, -14), textcoords="offset points", color="white")
b = HILL["bottom"]; ax.plot(*b, "*", color="white", ms=15, zorder=6)
ax.annotate("bottom (-3.19, 8.99), loss 2.8568", b, xytext=(-40, -22), textcoords="offset points", color="white")
ax.set_xlabel("mat's first number"); ax.set_ylabel("mat's second number"); ax.legend(loc="lower left", fontsize=9)
ax.set_title("Signature 6: the Loss Hill, a long flat river valley")
save(fig, os.path.join(OUT, "fig17-05-loss-hill.png"))

# 06 the three-node network, forward values above, backward nudges below
fig, ax = canvas(10.0, 4.0)
nodes = [("x = 2", NAVY), ("z = w1 x + b1\n= 0.5", BLUE), ("h = max(0, z)\n= 0.5", TEAL), ("y = w2 h + b2\n= 0.75", BLUE), ("loss (y - 2)²\n= 1.5625", VERM)]
for i, (t, col) in enumerate(nodes):
    box(ax, 2 + 20 * i, 16, 16, 11, t, color=col, fs=9.5)
    if i: arrow(ax, 20 * i - 2, 21.5, 20 * i + 2, 21.5)
back = ["dw1 = -7.5\ndb1 = -3.75", "dz = -3.75\n(ReLU on: × 1)", "dh = -3.75\n(= dy × w2)", "dy = -2.5\ndw2 = -1.25, db2 = -2.5"]
for i, t in enumerate(back):
    label(ax, 10 + 20 * i, 8, t, color=PLUM, fs=9.5)
label(ax, 50, 36, "Forward: values flow right.  Backward: nudges flow left, multiplied at every step", color=NAVY, fs=12, weight="bold")
arrow(ax, 92, 3, 8, 3, color=PLUM, lw=1.2)
save(fig, os.path.join(OUT, "fig17-06-three-node.png"))

# 07 the nudge each seat's row receives in the hand Tiny
rows = np.array(HN["rows"]); gX = np.array(HN["seat_nudges"])
names = ["the@0", "cat@1", "sat@2", "on@3", "the@4", "mat@5"]
fig, ax = plt.subplots(figsize=(7.6, 6.0))
k = 8.0
for i in range(6):
    ax.plot(*rows[i], "o", color=SERIES[i % len(SERIES)], ms=8)
    ax.annotate(names[i], rows[i], xytext=(6, 6), textcoords="offset points", fontsize=10)
    if np.linalg.norm(gX[i]) > 0:
        ax.annotate("", xy=rows[i] - k * gX[i], xytext=rows[i], arrowprops=dict(arrowstyle="-|>", color=SERIES[i % len(SERIES)], lw=1.6))
ax.set_xlabel("first number"); ax.set_ylabel("second number"); ax.set_aspect("equal", adjustable="datalim")
ax.set_title("Each seat's row and the way downhill (minus the nudge, drawn 8 times longer)")
save(fig, os.path.join(OUT, "fig17-07-seat-nudges.png"))

# 08 training the hand Tiny: wiggle against autograd, and four learning rates
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.2))
a1.plot(TH["wiggle"], color=AMBER, lw=4, alpha=0.6, label=f"nudges by wiggling ({TH['seconds']['wiggle']:.1f} s)")
a1.plot(TH["torch"], color=NAVY, lw=1.4, label=f"nudges by autograd ({TH['seconds']['torch']:.1f} s)")
a1.set_yscale("log"); a1.set_xlabel("step"); a1.set_ylabel("loss"); a1.legend(); a1.set_title("Two ways to nudge, one curve (lr 0.5)", fontsize=12)
for (lr, Ls), col in zip(TH["lrs"].items(), (TEAL, BLUE, AMBER, VERM)):
    Ls = np.array(Ls, float); Ls[~np.isfinite(Ls)] = np.nan
    a2.plot(Ls, color=col, label=f"lr {lr}")
a2.set_yscale("log"); a2.set_ylim(5e-3, 1e8); a2.set_xlabel("step"); a2.legend(); a2.set_title("Learning rates, hand Tiny, 200 steps", fontsize=12)
save(fig, os.path.join(OUT, "fig17-08-hand-training.png"))

# 09 training the real Tiny: held-out loss by step
fig, ax = plt.subplots(figsize=(9.0, 4.8))
for (lr, v), col in zip(TT.items(), (TEAL, BLUE, AMBER, PLUM, VERM)):
    c = np.array(v["curve"])
    if v["stopped"] < 600:
        ax.plot(np.arange(len(v["train"])), np.clip(v["train"], 0, 12), color=col, lw=1.2, ls=":", label=f"lr {lr}: blew up at step {v['stopped']} (batch loss)")
    else:
        ax.plot(c[:, 0], c[:, 3], color=col, label=f"lr {lr}: held-out {v['end'][1]:.3f}")
ax.axhline(6.1230, color=GREY, ls="--", lw=1.2); ax.text(605, 6.15, "frequencies only, 6.123", color=GREY, fontsize=9, ha="right", va="bottom")
ax.set_ylim(5, 12); ax.set_xlabel("step (1,024 predictions each)"); ax.set_ylabel("loss on held-out text")
ax.legend(fontsize=9); ax.set_title("Tiny's first 600 steps of plain gradient descent")
save(fig, os.path.join(OUT, "fig17-09-tiny-training.png"))

# 10 what backward costs
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.0))
t = COST["table"]
a1.bar(["forward\nY = XW", "backward\nnudge on X", "backward\nnudge on W"], [t["f"], t["bx"], t["bw"]], color=[BLUE, PLUM, PLUM])
a1.set_ylabel("ms, one thread"); a1.set_title("One table product: forward 2, backward 2 + 2", fontsize=12)
labels = list(COST["tiny"]); fw = [COST["tiny"][k]["forward_ms"] for k in labels]; bw = [COST["tiny"][k]["fwd_bwd_ms"] - COST["tiny"][k]["forward_ms"] for k in labels]
xx = np.arange(len(labels))
a2.bar(xx - 0.18, fw, 0.36, color=BLUE, label="forward"); a2.bar(xx + 0.18, bw, 0.36, color=PLUM, label="backward")
a2.set_xticks(xx, labels); a2.set_ylabel("ms, 1,024 tokens, one thread"); a2.legend()
a2.set_title("Tiny: backward 1.46 times forward (32-bit), 3.43 (64-bit)", fontsize=12)
save(fig, os.path.join(OUT, "fig17-10-backward-cost.png"))

# 11 wiggle against autograd on the real Tiny
fig, ax = plt.subplots(figsize=(6.8, 6.0))
a = np.array([abs(r["autograd"]) for r in TC["rows"]]); w = np.array([abs(r["wiggle"]) for r in TC["rows"]])
ax.loglog(a, w, "o", color=NAVY, ms=7)
for r, ai, wi in zip(TC["rows"], a, w):
    ax.annotate(r["name"], (ai, wi), xytext=(5, -10), textcoords="offset points", fontsize=8, color=SLATE)
lim = [3e-5, 1.0]; ax.plot(lim, lim, color=GREY, lw=1, ls="--")
ax.set_xlabel("size of autograd's nudge"); ax.set_ylabel("size of the wiggled nudge")
ax.set_title("Twelve of Tiny's numbers: wiggle and autograd agree\n(largest relative difference 1.1e-05)", fontsize=12)
save(fig, os.path.join(OUT, "fig17-11-tiny-check.png"))
print("wrote", sorted(os.listdir(OUT)))
