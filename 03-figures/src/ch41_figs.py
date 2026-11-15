"""Figures for Chapter 41, Circuits in Tiny.  Numbers are read from 04-code/ch41/outputs/ (run 04-code/ch41/RUN.txt first).
Run from the project root: python3 03-figures/src/ch41_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
setup()
OUT = os.path.join(ROOT, "03-figures", "ch41"); os.makedirs(OUT, exist_ok=True)
O = os.path.join(ROOT, "04-code", "ch41", "outputs")
J = lambda n: json.load(open(os.path.join(O, n + ".json")))
p = lambda n: os.path.join(OUT, n)
L = list(range(13))

# 01 the logit lens and the tuned lens, layer by layer
R = J("lens"); lg, tu = R["logit"], R["tuned"]
fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.1))
ax = axs[0]
ax.plot(L, [r[0] for r in lg], color=VERM, marker="o", ms=3.5, label="logit lens (the model's own output table)")
ax.plot(L, [r[0] for r in tu], color=BLUE, marker="s", ms=3.5, label="tuned lens (one affine map per layer)")
ax.axhline(7.625, color=GREY, ls=":", lw=1); ax.text(6.2, 7.85, "uniform guess 7.625", fontsize=7.5, color=GREY)
ax.axhline(3.523, color=GREEN, ls="--", lw=1); ax.text(0.3, 3.0, "finished model 3.523", fontsize=7.5, color=GREEN)
ax.set_ylim(2.5, 12.5); ax.text(1.3, 12.0, "layer 0: 23.9 (off scale)", fontsize=7.5, color=VERM)
ax.set_xlabel("layer read"); ax.set_ylabel("loss on the true next token (nats)"); ax.set_xticks([0, 3, 6, 9, 12])
ax.set_title("What each layer would predict if the model stopped there", fontsize=9.5); ax.legend(fontsize=7.5, loc="upper right")
ax = axs[1]
ax.plot(L, [r[3] for r in lg], color=VERM, marker="o", ms=3.5, label="logit lens")
ax.plot(L, [r[3] for r in tu], color=BLUE, marker="s", ms=3.5, label="tuned lens")
ax.set_xlabel("layer read"); ax.set_ylabel("share of seats agreeing with the final top choice"); ax.set_ylim(0, 1.03); ax.set_xticks([0, 3, 6, 9, 12])
ax.set_title("Agreement with the finished model", fontsize=9.5); ax.legend(fontsize=7.5, loc="upper left")
fig.suptitle("Reading Tiny's rows through a lens (Chapter 24's Tiny, held-out windows)", fontsize=11, x=0.01, ha="left")
save(fig, p("fig41-01-lens.png"))

# 02 the sparse autoencoder sweep at layer 6
S = [J(f"sae_l1_{x}") for x in ("0.03", "0.1", "0.3")]
fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.1))
ax = axs[0]
ax.plot([s["l0"] for s in S], [s["ve"] for s in S], color=BLUE, marker="o", ms=6)
for s in S: ax.annotate(f"L1 {s['l1']}", (s["l0"], s["ve"]), textcoords="offset points", xytext=(6, -12), fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("active features per seat (L0, log scale)"); ax.set_ylabel("share of variance kept"); ax.set_ylim(0.5, 1.02)
ax.set_title("Sparsity against fidelity, 1,024 features", fontsize=9.5)
ax = axs[1]
xs = range(3); ax.bar(xs, [s["dead"] for s in S], color=[BLUE, AMBER, VERM])
for i, s in enumerate(S): ax.text(i, s["dead"] + 15, str(s["dead"]), ha="center", fontsize=8.5)
ax.set_xticks(list(xs)); ax.set_xticklabels([f"L1 {s['l1']}" for s in S]); ax.set_ylabel("features never active on the training rows"); ax.set_ylim(0, 1024)
ax.set_title("Dead features out of 1,024", fontsize=9.5)
fig.suptitle("A sparse dictionary for Tiny's layer-6 rows", fontsize=11, x=0.01, ha="left")
save(fig, p("fig41-02-sae.png"))
print("ok")
