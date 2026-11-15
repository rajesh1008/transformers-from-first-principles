"""Figures for Chapter 32, Speculative Decoding.  Numbers are read from 04-code/ch32/outputs/ (run 04-code/ch32/RUN.txt
first).  Run from the project root: python3 03-figures/src/ch32_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch32"); os.makedirs(OUT, exist_ok=True)
J = lambda n: json.load(open(os.path.join(ROOT, "04-code", "ch32", "outputs", n + ".json")))
p = lambda n: os.path.join(OUT, n)
SS, SC = J("spec_sample"), J("spec_cost")

# 01 expected tokens per pass against acceptance, k = 1, 2, 4, 8, with the break-even of a 1.62-step verify pass
fig, ax = plt.subplots(figsize=(7.6, 5.6))
a = np.linspace(0, 0.95, 96)
for k, col, ls in ((1, GREY, ":"), (2, TEAL, "--"), (4, BLUE, "-"), (8, PLUM, "-.")):
    ax.plot(a, (1 - a ** (k + 1)) / (1 - a), color=col, ls=ls)
    ax.annotate(f"k = {k}", (0.95, (1 - 0.95 ** (k + 1)) / 0.05), xytext=(4, 0), textcoords="offset points", color=col, fontsize=10)
ax.axhline(1.62, color=VERM, ls="--", lw=1)
ax.annotate("cost of a k = 4 verify pass here (1.62 steps)", (0.45, 1.62), xytext=(0, -16), textcoords="offset points", color=VERM, fontsize=10)
for lab, acc, tpp in (("' the' (measured)", 0.008, 1.03), ("early exit 6 (measured)", 0.104, 1.39), ("prompt lookup (measured)", 0.353, 2.31)):
    ax.plot([acc], [tpp], "o", color=NAVY)
    ax.annotate(lab, (acc, tpp), xytext=(-20, 10), textcoords="offset points", color=NAVY, fontsize=9)
ax.set_xlabel("chance each guess is kept (independent)"); ax.set_ylabel("tokens confirmed per target pass")
ax.set_title("Tokens per pass if guesses were independent; dots: measured at k = 4", fontsize=12)
save(fig, p("fig32-01-accept.png"))

# 02 early-exit acceptance against depth
fig, ax = plt.subplots(figsize=(7.6, 5.6))
L = [2, 4, 6, 8, 10]; F = SS["F"]
for key, col, mk, ls, lab in (("greedy", NAVY, "o", "-", "greedy agreement"), ("0.7", BLUE, "s", "--", "expected acceptance, T 0.7"),
                              ("1.0", PLUM, "^", ":", "expected acceptance, T 1.0")):
    ax.plot(L, [F[str(l)][key] for l in L], marker=mk, ls=ls, color=col, label=lab)
ax.set_xlabel("layers of GPT-2 used by the draft (of 12)"); ax.set_ylabel("share of guesses kept")
ax.set_ylim(0, 0.45); ax.legend(frameon=False, loc="upper left")
ax.set_title("Untrained early exits as drafts, 512 held-out positions", fontsize=12)
save(fig, p("fig32-02-exits.png"))

# 03 prompt lookup speed per prompt, greedy against sampling
E = SS["E"]; names = ["separator (token 0)", "held-out 800", "held-out 1600", "held-out 2400"]
fig, ax = plt.subplots(figsize=(7.6, 5.6))
x = np.arange(4); w = 0.26
for j, (tag, col) in enumerate((("greedy", BLUE), ("T 0.7", TEAL), ("T 1.0", PLUM))):
    ax.bar(x + (j - 1) * w, [E[f"{tag} | {n}"]["speedup"] for n in names], w, color=col, label=tag)
ax.axhline(1.0, color=VERM, ls="--", lw=1); ax.annotate("plain decoding", (3.35, 1.0), xytext=(-40, 5), textcoords="offset points", color=VERM, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(["separator", "held-out 800", "held-out 1,600", "held-out 2,400"])
ax.set_ylabel("speed relative to plain decoding (x)"); ax.legend(frameon=False)
ax.set_title("Prompt lookup, k = 4, 64 new tokens per prompt", fontsize=12)
save(fig, p("fig32-03-lookup.png"))

# 04 cost of a five-row verify against one row, by batch size
H = SC["H"]; Bs = [1, 4, 16]
fig, ax = plt.subplots(figsize=(7.6, 5.6))
ax.plot(Bs, [H[f"B{b} r5"] / H[f"B{b} r1"] for b in Bs], "o-", color=VERM)
for b in Bs:
    r = H[f"B{b} r5"] / H[f"B{b} r1"]
    ax.annotate(f"{r:.2f}x", (b, r), xytext=(6, -12), textcoords="offset points", color=VERM, fontsize=10)
ax.set_xscale("log", base=2); ax.set_xticks(Bs); ax.set_xticklabels([str(b) for b in Bs])
ax.set_xlabel("conversations per decode step (B)"); ax.set_ylabel("five rows per conversation / one row (time)")
ax.set_ylim(1, 4); ax.set_title("GPT-2 small, context 128, one CPU thread: the slack speculation spends", fontsize=12)
save(fig, p("fig32-04-batch.png"))
