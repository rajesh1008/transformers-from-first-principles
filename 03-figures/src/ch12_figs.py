"""Figures for Chapter 12, The Feed-Forward Block as Memory.  Every number is read from 04-code/ch12/outputs or
recomputed here from tiny/ffn.py.  Run from the project root: python3 03-figures/src/ch12_figs.py"""
import os, sys, re, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code"))
from style import *
import matplotlib.pyplot as plt
from tiny.ffn import relu, gelu, silu
setup()
OUT = os.path.join(ROOT, "03-figures", "ch12"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch12", "outputs")

# 01 the three activations
x = np.linspace(-4, 3, 701)
fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.plot(x, relu(x), color=NAVY, label="ReLU"); ax.plot(x, gelu(x), color=TEAL, label="GELU"); ax.plot(x, silu(x), color=PLUM, label="SiLU")
ax.axhline(0, color="#999999", lw=0.8); ax.axvline(0, color="#999999", lw=0.8)
for f, c, nm in [(gelu, TEAL, "GELU"), (silu, PLUM, "SiLU")]:
    i = np.argmin(f(x)); ax.plot(x[i], f(x)[i], "o", color=c)
    ax.annotate(f"{nm} lowest {f(x)[i]:.3f} at {x[i]:.2f}", (x[i], f(x)[i]), (x[i] - 2.6, f(x)[i] - 0.65 + (0.25 if nm == 'GELU' else 0)),
                fontsize=9.5, color=c, arrowprops=dict(arrowstyle="-|>", color=c))
ax.set_ylim(-1.2, 3); ax.set_xlabel("match (input to the bend)"); ax.set_ylabel("activation"); ax.legend(loc="upper left")
ax.set_title("Three bends: ReLU cuts, GELU and SiLU dip below zero, then fade")
save(fig, os.path.join(OUT, "fig12-01-activations.png"))

# 02 key against value cosine by layer
txt = open(os.path.join(CH, "gpt2_neurons.txt")).read()
rows = re.findall(r"^\s+(\d+) \|\s+([-\d.]+) \|\s+([\d.]+) \|\s+([\d.]+) \|", txt, re.M)
L = [int(r[0]) for r in rows]; med = [float(r[1]) for r in rows]; hi = [float(r[2]) for r in rows]; lo = [float(r[3]) for r in rows]
fig, ax = plt.subplots(figsize=(8.4, 4.4)); w = 0.38; xx = np.arange(len(L))
ax.bar(xx - w / 2, lo, w, color=TINT[VERM], edgecolor=VERM, label="share of neurons with cosine below -0.5")
ax.bar(xx + w / 2, hi, w, color=TINT[TEAL], edgecolor=TEAL, label="share above 0.5")
ax2 = ax.twinx(); ax2.plot(xx, med, marker="o", color=NAVY, label="median cosine"); ax2.set_ylim(-0.6, 0.3); ax2.grid(False); ax2.set_ylabel("median cosine")
ax.set_xticks(xx, L); ax.set_xlabel("GPT-2 small layer"); ax.set_ylabel("share of 3,072 neurons"); ax.set_ylim(0, 0.6)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper right")
ax.set_title("Does a neuron write what it reads? Mostly the opposite, early on")
save(fig, os.path.join(OUT, "fig12-02-cosine.png"))

# 03 removing each block
ab = re.findall(r"layer\s+(\d+): Alice ([-+\d.]+) \| Looking-Glass ([-+\d.]+)", txt)
a = [float(r[1]) for r in ab]; g = [float(r[2]) for r in ab]
fig, ax = plt.subplots(figsize=(8.4, 4.4)); xx = np.arange(len(ab))
ax.bar(xx - w / 2, a, w, color=TINT[BLUE], edgecolor=BLUE, label="Alice's Adventures in Wonderland")
ax.bar(xx + w / 2, g, w, color=TINT[AMBER], edgecolor=AMBER, label="Through the Looking-Glass")
ax.set_ylim(-0.3, 0.4); ax.axhline(0, color="#999999", lw=0.8)
ax.text(0.6, 0.36, f"layer 0 off the scale: +{a[0]:.2f} and +{g[0]:.2f}", fontsize=10, color=VERM, va="center")
ax.set_xticks(xx, [r[0] for r in ab]); ax.set_xlabel("layer whose feed-forward block is removed"); ax.set_ylabel("change in loss")
ax.legend(loc="lower right"); ax.set_title("GPT-2 small without one feed-forward block, 1,024 tokens each")
save(fig, os.path.join(OUT, "fig12-03-ablation.png"))

# 04 the bracket neuron along Alice
act, inside = np.load(os.path.join(CH, "bracket_act.npy"))
n = 400
fig, ax = plt.subplots(figsize=(8.4, 4.0))
ax.fill_between(np.arange(n), -0.5, 5.5, where=inside[:n] > 0.5, color=TINT[PLUM], step="mid", label="inside a bracket")
ax.plot(np.arange(n), act[:n], color=NAVY, lw=1.1, label="neuron 10.888")
ax.set_ylim(-0.5, 5.5); ax.set_xlim(0, n - 1); ax.set_xlabel("token position in Alice"); ax.set_ylabel("activation")
ax.legend(loc="upper right"); ax.set_title("Neuron 10.888 rises inside brackets: first 400 tokens")
save(fig, os.path.join(OUT, "fig12-04-bracket.png"))
print("4 figures in", OUT)
