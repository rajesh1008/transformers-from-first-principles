"""Figures for Chapter 18, Backpropagation by Hand Through One Head of Attention.  Every number is read from
04-code/ch18/outputs/*.json (run the chapter's scripts first, in the order of 04-code/ch18/RUN.txt).
Run from the project root:  python3 03-figures/src/ch18_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch18"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch18", "outputs")
js = lambda n: json.load(open(os.path.join(CH, n + ".json")))
RC, AR, BH, AG, SAT, BF, MEM, COST = (js(n) for n in ("recap", "arriving", "backward_head", "autograd30", "saturation", "bf16_leak", "memory", "cost"))
G = {k: np.array(v) for k, v in BH["g"].items()}
SEATS = ["the@0", "cat@1", "sat@2", "on@3", "the@4"]
P = np.array(RC["P"])
p = lambda n: os.path.join(OUT, n)

# 01 the map: forward left to right, nudges right to left
fig, ax = canvas(12, 4.6)
names = [("A", "rows read", AMBER), ("Q K V", "A times W_q, W_k, W_v", NAVY), ("S", "Q Kᵀ / √2", BLUE),
         ("P", "softmax by row", PLUM), ("O", "P V", PLUM), ("Y", "O W_o", NAVY)]
for i, (t, s_, col) in enumerate(names):
    box(ax, 2 + i * 19.6, 24, 15, 10, t, col, sub=s_, fs=13 if len(s_) < 18 else 12)
    if i:
        arrow(ax, 2 + i * 19.6 - 4.2, 31, 2 + i * 19.6 - 0.4, 31, color=GREY)
        arrow(ax, 2 + i * 19.6 - 0.4, 21, 2 + i * 19.6 - 4.2, 21, color=VERM)
rules = ["dA = dQ W_qᵀ\n+ dK W_kᵀ + dV W_vᵀ", "dQ = c dS K\ndK = c dSᵀ Q", "dS = P ⊙ (dP\n− rowsum(P ⊙ dP))",
         "dP = dO Vᵀ\ndV = Pᵀ dO", "dO = dY W_oᵀ\ndW_o = Oᵀ dY"]
for i, r in enumerate(rules):
    label(ax, 2 + (i + 1) * 19.6 - 2.3, 12.5, r, color=VERM, fs=9)
label(ax, 60, 42, "forward: each table is made from the one to its left", color=GREY, fs=10.5)
label(ax, 60, 4, "backward: each nudge is made from the one to its right, and from what the forward pass kept", color=VERM, fs=10.5)
save(fig, p("fig18-01-backward-map.png"))

# 02 dY arriving, autograd and wiggle
dY = np.array(AR["dY"]); wg = np.array(AR["wig"])
fig, ax = plt.subplots(figsize=(8.6, 4.2))
x = np.arange(5); w = 0.2
for j, (col, nm) in enumerate([(BLUE, "first number"), (TEAL, "second number")]):
    ax.bar(x + (j - 0.5) * 2 * w - w / 2, dY[:, j], w, color=col, label=f"autograd, {nm}")
    ax.bar(x + (j - 0.5) * 2 * w + w / 2, wg[:, j], w, color=col, alpha=0.45, hatch="//", label=f"wiggled, {nm}")
ax.axhline(0, color=SLATE, lw=0.8); ax.set_xticks(x, SEATS); ax.set_ylabel("nudge on the head's output, dY")
ax.set_title("The nudge that arrives at layer 0's head, checked by wiggling (largest difference 4e-10)")
ax.legend(fontsize=8.5, ncol=2)
save(fig, p("fig18-02-arriving.png"))

# 03 the backward table: dS with masked cells and row sums
fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw=dict(width_ratios=[1, 1]))
for ax, T, ttl, cmap in [(axs[0], np.tril(P[:, :5]), "shares P (forward; each row adds to 1)", "Purples"), (axs[1], G["S"], "score nudges dS (backward)", "RdBu_r")]:
    M = np.where(np.tril(np.ones((5, 5))) > 0, T, np.nan)
    lim = np.nanmax(np.abs(M)) if cmap == "RdBu_r" else None
    im = ax.imshow(M, cmap=cmap, vmin=-lim if lim else 0, vmax=lim if lim else 1)
    for i in range(5):
        for j in range(5):
            dark = j <= i and abs(M[i, j]) > 0.6 * (lim or 1)
            ax.text(j, i, f"{M[i, j]:+.4f}".replace("-", "−") if cmap == "RdBu_r" and j <= i else (f"{M[i, j]:.3f}" if j <= i else ""),
                    ha="center", va="center", fontsize=8, color="white" if dark else SLATE)
    ax.set_xticks(range(5), SEATS, fontsize=8); ax.set_yticks(range(5), SEATS, fontsize=8); ax.grid(False)
    ax.set_title(ttl, fontsize=11)
for i in range(5):
    axs[1].text(4.9, i, f"  row sum {abs(G['S'][i].sum()):.0e}", va="center", fontsize=8, color=VERM)
save(fig, p("fig18-05-backward-table.png"))

# 04 softmax slopes for sat@2
s = np.array(RC["S"][2][:3]); pp = P[2, :3]; J = np.diag(pp) - np.outer(pp, pp)
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9))
axs[0].bar(range(3), pp, color=PLUM); axs[0].set_xticks(range(3), SEATS[:3]); axs[0].set_ylim(0, 0.5)
for i, v in enumerate(pp):
    axs[0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)
axs[0].set_title("sat@2's three shares", fontsize=11)
im = axs[1].imshow(J, cmap="RdBu_r", vmin=-0.25, vmax=0.25); axs[1].grid(False)
for i in range(3):
    for j in range(3):
        axs[1].text(j, i, f"{J[i, j]:+.4f}", ha="center", va="center", fontsize=9)
axs[1].set_xticks(range(3), ["score " + s_ for s_ in SEATS[:3]], fontsize=8); axs[1].set_yticks(range(3), ["share " + s_ for s_ in SEATS[:3]], fontsize=8)
axs[1].set_title("slope of each share in each score: P_i([i=j] - P_j)", fontsize=11)
save(fig, p("fig18-04-softmax-slopes.png"))

# 05 one key, four readers
Q = np.array(RC["Q"]); c = 1 / np.sqrt(2)
parts = [G["S"][i, 1] * Q[i] * c for i in range(1, 5)]
fig, ax = plt.subplots(figsize=(7.6, 5.6))
o = np.zeros(2); cols = [BLUE, TEAL, AMBER, PLUM]
offs = [(-0.0019, -0.0004), (-0.0012, 0.0007), (0.0003, 0.0), (0.0004, 0.0)]
for pt, col, i, of in zip(parts, cols, range(1, 5), offs):
    ax.annotate("", xy=o + pt, xytext=o, arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
    ax.text(*(o + pt / 2 + np.array(of)), f"from {SEATS[i]}", color=col, fontsize=9); o = o + pt
ax.annotate("", xy=o, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=VERM, lw=2.6, ls="--"))
ax.text(o[0], o[1], f"  sum = dK[cat@1]\n  ({o[0]:+.5f}, {o[1]:+.5f})", color=VERM, fontsize=9, va="top")
ax.set_xlim(-0.004, 0.008); ax.set_ylim(-0.011, 0.003); ax.set_aspect("equal")
ax.set_xticks([-0.004, 0, 0.004, 0.008]); ax.set_yticks([-0.008, -0.004, 0])
ax.set_xlabel("first number"); ax.set_ylabel("second number")
ax.set_title("cat@1's key is read by four seats: its nudge is the sum")
save(fig, p("fig18-03-four-readers.png"))

# 06 three paths into A
parts = {k: np.array(v) for k, v in BH["parts"].items()}
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
for d, ax in enumerate(axs):
    bottom_p = np.zeros(5); bottom_n = np.zeros(5)
    for k_, col in (("q", BLUE), ("k", TEAL), ("v", PLUM)):
        v = parts[k_][:, d]
        ax.bar(range(5), np.where(v > 0, v, 0), bottom=bottom_p, color=col, label=f"via {k_.upper()}")
        ax.bar(range(5), np.where(v < 0, v, 0), bottom=bottom_n, color=col)
        bottom_p += np.where(v > 0, v, 0); bottom_n += np.where(v < 0, v, 0)
    ax.plot(range(5), G["A"][:, d], "o", color=VERM, label="total dA")
    ax.axhline(0, color=SLATE, lw=0.8); ax.set_xticks(range(5), SEATS, fontsize=9)
    ax.set_title(f"dA, {'first' if d == 0 else 'second'} number", fontsize=11)
axs[0].legend(fontsize=8.5)
save(fig, p("fig18-06-three-paths.png"))

# 07 skip plus head equals Chapter 17's seat nudges
dX = np.array(BH["dX"]); th = np.array(BH["through"])
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
for d, ax in enumerate(axs):
    ax.bar(np.arange(5) - 0.2, dY[:, d], 0.4, color=SLATEBLUE, label="skip path (dY)")
    ax.bar(np.arange(5) + 0.2, th[:, d], 0.4, color=NAVY, label="through the head and ln_1")
    ax.plot(range(5), dX[:, d], "D", color=VERM, label="sum = Chapter 17's seat nudge")
    ax.axhline(0, color=SLATE, lw=0.8); ax.set_xticks(range(5), SEATS, fontsize=9)
    ax.set_title(f"nudge on the row entering layer 0, {'first' if d == 0 else 'second'} number", fontsize=10.5)
axs[0].legend(fontsize=8.5)
save(fig, p("fig18-07-seat-nudges.png"))

# 08 agreement ladder
lad = BH["ladder"]
fig, ax = plt.subplots(figsize=(8.6, 3.8))
vals = [max(d, 1e-20) for _, d in lad]
ax.bar(range(len(lad)), vals, color=[TEAL if d > 0 else GREEN for _, d in lad]); ax.set_yscale("log")
ax.set_xticks(range(len(lad)), [n.replace(" (seen seats)", "") for n, _ in lad], fontsize=9)
ax.axhline(2.2e-16, color=GREY, ls="--", lw=1); ax.text(len(lad) - 0.5, 3e-16, "64-bit rounding step near 1", color=GREY, fontsize=8, ha="right")
for i, (_, d) in enumerate(lad):
    ax.text(i, max(d, 1e-20) * 1.6, "0" if d == 0 else f"{d:.0e}", ha="center", fontsize=8)
ax.set_ylim(1e-20, 1e-14); ax.set_ylabel("largest difference from autograd")
ax.set_title("Every step by hand against autograd, in the order the nudges travel")
save(fig, p("fig18-09-agreement.png"))

# 09 the running sentence with its nudges beneath
fig, ax = canvas(10, 4.4)
words = ["the", "cat", "sat", "on", "the", "mat"]
for i, wd in enumerate(words):
    x0 = 3 + i * 16
    box(ax, x0, 31, 12, 8, wd + f"@{i}", AMBER, fs=12)
    if i < 5:
        for r, (k_, col) in enumerate((("Q", BLUE), ("K", TEAL), ("V", PLUM))):
            v = G[k_][i]; nrm = np.linalg.norm(v)
            y0 = 22 - r * 7.5
            ax.add_patch(plt.Rectangle((x0, y0), max(np.sqrt(nrm / 0.0601) * 12, 0.15 if nrm else 0), 3, color=col))
            label(ax, x0 - 0.6, y0 + 1.5, f"d{k_}", color=col, fs=8, ha="right")
            label(ax, x0, y0 + 4.4, "exactly 0" if nrm == 0 else f"{nrm:.4f}", color=SLATE, fs=7.5, ha="left")
    else:
        label(ax, x0 + 6, 18, "no loss reads it:\nevery nudge 0", color=GREY, fs=8.5)
label(ax, 50, 3, "bar length: square root of the size of each seat's query, key and value nudge in layer 0 (full bar = 0.0601)", color=GREY, fs=9)
save(fig, p("fig18-08-sentence-nudges.png"))

# 10 saturation
fig, ax = plt.subplots(figsize=(8.6, 4.2))
fa = [r["factor"] for r in SAT]
ax.plot(fa, [r["dS"] for r in SAT], "o-", color=VERM, label="size of the nudge reaching the scores")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("sharpness factor on the@4's scores")
ax.set_ylabel("size of dS"); ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(fa, [r["top"] for r in SAT], "s--", color=PLUM, label="largest share"); ax2.set_ylim(0, 1.05); ax2.set_ylabel("largest share", color=PLUM)
ax.legend(loc="lower left", fontsize=9); ax2.legend(loc="center left", fontsize=9)
ax.set_title("Hard shares pass back almost nothing")
save(fig, p("fig18-12-saturation.png"))

# 11 bf16 leak
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
four = BF["four"]
axs[0].bar([0, 1, 2], [0, four["leak"][0], four["proj"][0]], color=[GREEN, CORAL, TEAL])
axs[0].set_xticks([0, 1, 2], ["exact", "dS rounded\nto bfloat16", "rounded, row\nmean removed"], fontsize=9)
for i, v in enumerate([0, four["leak"][0], four["proj"][0]]):
    axs[0].text(i, v - 1.5 if v < 0 else 0.6, f"{v:+.1f}".replace("-", "\u2212"), ha="center", fontsize=10)
axs[0].set_ylim(-28, 4); axs[0].set_title("four keys: first number of dQ", fontsize=11)
R = BF["random"]; off = [max(r["offset"], 0.25) for r in R]
axs[1].plot(off, [r["rounded"] for r in R], "o-", color=CORAL, label="dS rounded to bfloat16")
axs[1].plot(off, [r["removed"] for r in R], "s-", color=TEAL, label="rounded, row mean removed")
axs[1].set_xscale("log", base=2); axs[1].set_yscale("log"); axs[1].legend(fontsize=9)
axs[1].set_xlabel("shared key offset (0 plotted at 1/4)"); axs[1].set_ylabel("relative error in dQ")
axs[1].set_title("random head, 64 keys: error grows with the offset", fontsize=11)
save(fig, p("fig18-13-bf16-leak.png"))

# 12 memory
fig, ax = plt.subplots(figsize=(8.6, 4))
ax.barh([1], [MEM["tables_mb"]], color=NAVY, label="learned tables")
ax.barh([0], [MEM["tables_mb"]], color=NAVY)
ax.barh([0], [MEM["grads_mb"]], left=[MEM["tables_mb"]], color=TEAL, label="their nudges")
ax.barh([0], [MEM["share_mb"]], left=[2 * MEM["tables_mb"]], color=PLUM, label="kept: 1,024 x 1,024 tables")
ax.barh([0], [MEM["saved_mb"] - MEM["share_mb"]], left=[2 * MEM["tables_mb"] + MEM["share_mb"]], color=SLATEBLUE, label="kept: everything else")
ax.set_yticks([0, 1], ["training step", "inference (no_grad)"]); ax.set_xlabel("MB, 32-bit, Tiny on 1,024 tokens")
tot = 2 * MEM["tables_mb"] + MEM["saved_mb"]
ax.text(tot + 5, 0, f"{tot:.0f} MB", va="center"); ax.text(MEM["tables_mb"] + 5, 1, f"{MEM['tables_mb']:.1f} MB", va="center")
ax.set_xlim(0, tot * 1.15); ax.legend(fontsize=8.5, loc="upper right")
ax.set_title("What a training step holds that inference does not")
save(fig, p("fig18-10-memory.png"))

# 13 cost
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
axs[0].bar(["forward", "backward", "tiled\nbackward"], [2, 4, 5], color=[BLUE, VERM, TEAL])
axs[0].set_ylabel("n x n x d products"); axs[0].set_title("products per head", fontsize=11)
axs[1].bar(["products,\nforward", "products,\nbackward", "whole\nforward", "dense\nbackward", "tiled\nbackward"],
           [COST["pf"], COST["pb"], COST["tf"], COST["tb"], COST["tt"]], color=[BLUE, VERM, BLUE, VERM, TEAL])
for i, v in enumerate([COST["pf"], COST["pb"], COST["tf"], COST["tb"], COST["tt"]]):
    axs[1].text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
axs[1].set_ylabel("ms, one thread"); axs[1].set_title(f"one head, 1,024 seats: kept {COST['kept_dense']/1e6:.2f} MB dense, {COST['kept_tiled']/1e6:.2f} MB tiled", fontsize=10.5)
axs[1].tick_params(axis="x", labelsize=8.5)
save(fig, p("fig18-11-cost.png"))
print("13 figures written to", OUT)
