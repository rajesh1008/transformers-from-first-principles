"""Figures for Chapter 45, Distributed Training on Paper.  Pure arithmetic, the same formulas as 04-code/ch45/paper45.py
(checked by simulation in lab45.py).  Run from the project root: python3 03-figures/src/ch45_figs.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt, numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch45"); os.makedirs(OUT, exist_ok=True); p = lambda n: os.path.join(OUT, n)
# 01 pipeline bubble against micro-batches
fig, ax = plt.subplots(figsize=(9.0, 4.8)); m = np.arange(1, 129)
for st, col in ((4, TEAL), (8, BLUE), (16, VERM)):
    ax.plot(m, (st - 1) / (m + st - 1), color=col, lw=2.4, label=f"{st} stages")
ax.axhline(0.1, color=GREY, ls="--", lw=1.2); ax.text(70, 0.11, "10 per cent idle", color=GREY, fontsize=9)
ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
ax.set_xlabel("micro-batches per step (m)"); ax.set_ylabel("idle share of the schedule"); ax.set_ylim(0, 1); ax.legend(fontsize=9)
fig.suptitle("The pipeline bubble, (p - 1) / (m + p - 1)", fontsize=11, x=0.01, ha="left"); save(fig, p("fig45-01-bubble.png"))
# 02 memory per rank for an illustrative 70B dense model under ZeRO stages
N = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256]); P = 70e9
fig, ax = plt.subplots(figsize=(9.0, 4.8))
for lab, f, col in (("plain data parallel", lambda n: 16 * P + 0 * n, GREY), ("ZeRO-1: optimiser state", lambda n: (4 + 12 / n) * P, AMBER),
                    ("ZeRO-2: + gradients", lambda n: (2 + 14 / n) * P, PLUM), ("ZeRO-3 / FSDP: + weights", lambda n: 16 * P / n, TEAL)):
    ax.plot(N, f(N) / 1e9, "o-", color=col, lw=2.2, label=lab)
ax.axhline(80, color=VERM, ls="--", lw=1.4); ax.text(1.1, 90, "80 GB accelerator", color=VERM, fontsize=9)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(N); ax.set_xticklabels([str(n) for n in N])
ax.set_xlabel("data-parallel ranks"); ax.set_ylabel("training state per rank (GB)"); ax.legend(fontsize=9, loc="lower left")
fig.suptitle("Training state per rank for a 70-billion-number model, 16 bytes per learned number", fontsize=11, x=0.01, ha="left"); save(fig, p("fig45-02-zero.png"))
print("ok")
