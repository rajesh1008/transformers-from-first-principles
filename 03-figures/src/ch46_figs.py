"""Figures for Chapter 46, Common Misconceptions, Corrected.  Numbers read from 04-code/ch46/outputs/lab46.json
(fig46-01) and from Chapter 40's measured head ablations (fig46-02: 0.08, 0.29, 3.48 nats, 04-code/ch40/outputs).
Run from the project root: python3 03-figures/src/ch46_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt, numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch46"); os.makedirs(OUT, exist_ok=True); p = lambda n: os.path.join(OUT, n)
R = json.load(open(os.path.join(ROOT, "04-code", "ch46", "outputs", "lab46.json")))
# 01 the same 64 tokens, read with longer contexts
L = [64, 128, 256, 512]; y = [R["F"][str(l)] for l in L]
fig, ax = plt.subplots(figsize=(9.0, 4.8))
bars = ax.bar([str(l) for l in L], y, color=[TEAL, AMBER, AMBER, VERM], width=0.6)
for b, v in zip(bars, y): ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.4f}", ha="center", fontsize=10)
ax.set_ylim(4.0, 5.5); ax.set_xlabel("tokens of context before the 64 scored tokens"); ax.set_ylabel("loss on the scored tokens (nats per token)")
ax.text(-0.35, 5.38, "Tiny trained on windows of 128 positions. Whole window read: 128, 192, 320, 576.", ha="left", fontsize=9, color=GREY)
fig.suptitle("More context made Tiny worse on the same 64 Looking-Glass tokens (8 places)", fontsize=11, x=0.01, ha="left"); save(fig, p("fig46-01-context.png"))
# 02 attention ranking against intervention (Chapter 40)
lab = ["four heads with the most\nattention on the relevant token", "four random heads", "four heads whose removal\ncost most one at a time"]
v = [0.08, 0.29, 3.48]
fig, ax = plt.subplots(figsize=(9.0, 4.2))
bars = ax.barh(lab[::-1], v[::-1], color=[TEAL, GREY, BLUE], height=0.55)
for b, x in zip(bars, v[::-1]): ax.text(x + 0.05, b.get_y() + b.get_height() / 2, f"{x:.2f}", va="center", fontsize=10)
ax.set_xlabel("rise in loss when the four heads are zeroed together (nats)"); ax.set_xlim(0, 4.0)
fig.suptitle("Where Tiny's heads look is not what its answer depends on (Chapter 40)", fontsize=11, x=0.01, ha="left"); save(fig, p("fig46-02-attention.png"))
print("ok")
