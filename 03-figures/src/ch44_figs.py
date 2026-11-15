"""Figures for Chapter 44, Reasoning Models, Test-Time Compute and State-Space Models.  Numbers are read from
04-code/ch44/outputs/ (run 04-code/ch44/RUN.txt first).  Run from the project root: python3 03-figures/src/ch44_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt, numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch44"); os.makedirs(OUT, exist_ok=True)
O = os.path.join(ROOT, "04-code", "ch44", "outputs"); J = lambda n: json.load(open(os.path.join(O, n + ".json")))
p = lambda n: os.path.join(OUT, n)

# 01 four-digit addition: exact on 200 held-out pairs against step
fig, ax = plt.subplots(figsize=(9.0, 4.8))
for name, col, lab in (("pad4", TEAL, "scratchpad, 1,500 steps"), ("direct4", AMBER, "direct, 1,500 steps"), ("direct4_long", BLUE, "direct, 4,500 steps")):
    L = J(name)["log"]; ax.plot([r["step"] for r in L], [r["held200"] for r in L], "o-", color=col, lw=2.2, label=f"{lab} (final {J(name)['test']:.4f} on 2,000)")
ax.axvline(1500, color=GREY, ls="--", lw=1.2); ax.text(1530, 0.55, "short runs stop", color=GREY, fontsize=9)
ax.set_xlabel("training step"); ax.set_ylabel("exact on 200 held-out sums"); ax.set_ylim(-0.03, 1.05)
ax.legend(fontsize=9, loc="center right"); fig.suptitle("Four-digit addition: the scratchpad is learnt in 500 steps, the direct answer after a jump near 3,000", fontsize=11, x=0.01, ha="left")
save(fig, p("fig44-01-addition.png"))

# 02 memory carried while generating: Tiny's cache, a 1-in-4 hybrid, a state-space stack
n = np.logspace(1, np.log10(131072), 200); per_tok = 2 * 12 * 128 * 4; state = 12 * 256 * 19 * 4
fig, ax = plt.subplots(figsize=(9.0, 4.8))
ax.plot(n, per_tok * n, color=VERM, lw=2.4, label="Tiny, 12 attention layers: 12,288 bytes per token")
ax.plot(n, 3 * 1024 * n + 9 * 256 * 19 * 4, color=PLUM, lw=2.4, label="hybrid, 3 attention + 9 state-space layers")
ax.plot(n, np.full_like(n, state), color=TEAL, lw=2.4, label=f"12 state-space layers: {state:,} bytes at any length")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("tokens in the sequence"); ax.set_ylabel("bytes carried per sequence (32-bit)")
ax.legend(fontsize=9, loc="upper left"); fig.suptitle("What a sequence costs in memory while generating, for Tiny's shape", fontsize=11, x=0.01, ha="left")
save(fig, p("fig44-02-memory.png"))
print("ok")
