"""Figures for Chapter 34, Distillation and Pruning.  Numbers are read from 04-code/ch34/outputs/ (run 04-code/ch34/RUN.txt
first).  Run from the project root: python3 03-figures/src/ch34_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
setup()
OUT = os.path.join(ROOT, "03-figures", "ch34"); os.makedirs(OUT, exist_ok=True)
J = lambda n: json.load(open(os.path.join(ROOT, "04-code", "ch34", "outputs", n + ".json")))
p = lambda n: os.path.join(OUT, n)

# 01 distilling Tiny into four layers: held-out curves for the four losses
fig, ax = plt.subplots(figsize=(7.6, 5.2))
for mode, col, lab in (("hard", GREY, "labels only (hard)"), ("kd1", BLUE, "teacher's table, T = 1"),
                       ("kd2", TEAL, "teacher's table, T = 2"), ("mix", VERM, "half and half")):
    c = J(f"distill_{mode}_4L_2000")["curve"]
    ax.plot([s for s, _ in c], [b for _, b in c], marker="o", ms=4, color=col, label=lab)
ax.axhline(1.9812, color=NAVY, ls="--", lw=1)
ax.annotate("teacher, 12 layers: 1.9812", (260, 1.9812), xytext=(0, -14), textcoords="offset points", color=NAVY, fontsize=10)
ax.set_xlabel("training step (8 windows of 129 tokens each)"); ax.set_ylabel("bits per character, held-out Carroll")
ax.set_ylim(1.94, 2.7); ax.legend(frameon=False, loc="upper right")
ax.set_title("A four-layer student, same seed, data and steps; only the loss differs", fontsize=12)
save(fig, p("fig34-01-distill.png"))

# 02 pruning GPT-2 without retraining: three rankings
A = J("prune_gpt2")["A"]; base = A["base"]; fr = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8]
fig, ax = plt.subplots(figsize=(7.6, 5.2))
for key, col, mk, lab in (("table", GREY, "s", "smallest |w| in the whole map"), ("channel", BLUE, "o", "smallest |w| per output channel"),
                          ("wanda", TEAL, "^", "|w| x input norm per output channel (Wanda)")):
    ax.plot([f * 100 for f in fr], [A[f"{key} {f}"] for f in fr], marker=mk, color=col, label=lab)
ax.plot([30, 50], [A["random 0.3"], A["random 0.5"]], marker="x", color=VERM, ls=":", label="random")
ax.axhline(base, color=NAVY, ls="--", lw=1)
ax.annotate(f"unpruned: {base:.3f}", (62, base), xytext=(0, -15), textcoords="offset points", color=NAVY, fontsize=10)
ax.set_xlabel("per cent of linear-map weights set to zero"); ax.set_ylabel("bits per character, held-out Carroll")
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.set_title("GPT-2 small, no retraining", fontsize=12)
save(fig, p("fig34-02-prune-gpt2.png"))

# 03 prune Tiny, then recover for 500 steps
rows = [("units50", "half the\nfeed-forward units"), ("units75", "three quarters of\nthe units"), ("w80", "80% of weights,\nper channel")]
fig, ax = plt.subplots(figsize=(8.4, 5.0))
x = range(len(rows)); w = 0.26
for i, (lab, col, f) in enumerate((("after pruning", VERM, lambda r: J(f"prune_distill_{r}_hard_500")["after_prune"]),
                                   ("+500 steps, labels", GREY, lambda r: J(f"prune_distill_{r}_hard_500")["final"]),
                                   ("+500 steps, teacher's table", BLUE, lambda r: J(f"prune_distill_{r}_kd1_500")["final"]))):
    vals = [f(r) for r, _ in rows]
    ax.bar([k + (i - 1) * w for k in x], vals, w, color=col, label=lab)
    for k, v in zip(x, vals): ax.text(k + (i - 1) * w, v + 0.012, f"{v:.3f}", ha="center", fontsize=8, color=SLATE, rotation=90 if v < 2.1 else 0)
ax.axhline(1.9812, color=NAVY, ls="--", lw=1, label="unpruned: 1.9812")
ctl = J("prune_distill_none_hard_500")["final"]
ax.axhline(ctl, color=GREEN, ls=":", lw=1.4, label=f"unpruned +500 steps, labels: {ctl:.4f}")
ax.set_xticks(list(x)); ax.set_xticklabels([l for _, l in rows], fontsize=9)
ax.set_ylim(1.9, 2.85); ax.set_ylabel("bits per character, held-out Carroll"); ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.set_title("Tiny pruned, then trained with the pruned weights held at zero", fontsize=12)
save(fig, p("fig34-03-recover.png"))
print("wrote", sorted(os.listdir(OUT)))
