"""Figures for Chapter 50, from 04-code/ch50/outputs/lab50.json and heads50.json.  Run from the project root:
python3 03-figures/src/ch50_figs.py.  fig50-01 the share of attention on the first token by depth in three generations of
model; fig50-02 the six-by-six attention table of the running sentence, averaged over the heads of the last layer."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt, numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch50"); os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(ROOT, "04-code", "ch50", "outputs", "lab50.json")))
C = [BLUE, TEAL, VERM]
fig, a = plt.subplots(figsize=(9, 4.6))
for r, c in zip(R, C):
    s = r["sink_by_layer"]; x = (np.arange(len(s)) + 1) / len(s)
    a.plot(x, s, marker="o", ms=3.5, lw=1.5, color=c, label=f"{r['model']} ({r['year']}, {r['layers']} layers), mean {r['sink_mean']:.4f}")
a.set_xlabel("depth (layer number divided by the number of layers)"); a.set_ylabel("share of attention on the first token")
a.set_ylim(0, 1); a.set_title("The first token takes most of the attention in all three generations"); a.legend(frameon=False, fontsize=9, loc="lower right")
save(fig, os.path.join(OUT, "fig50-01-sink.png"))
fig, axs = plt.subplots(1, 3, figsize=(12, 4.3), gridspec_kw=dict(wspace=0.35))
for a, r in zip(axs, R):
    T = np.array(r["tables"]["last"]); im = a.imshow(T, cmap="Blues", vmin=0, vmax=1)
    lab = [p.strip() for p in r["pieces"]]; a.set_xticks(range(6)); a.set_xticklabels(lab, fontsize=9); a.set_yticks(range(6)); a.set_yticklabels(lab, fontsize=9)
    for i in range(6):
        for j in range(i + 1):
            a.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if T[i, j] > 0.5 else NAVY)
    a.set_title(f"{r['model']} ({r['year']})", fontsize=11); a.set_xlabel("attended to"); a.set_ylabel("attending"); a.grid(False)
fig.suptitle("The running sentence's attention table, last layer, averaged over heads", fontsize=13, color=NAVY, weight="bold")
save(fig, os.path.join(OUT, "fig50-02-tables.png")); print("ok")
