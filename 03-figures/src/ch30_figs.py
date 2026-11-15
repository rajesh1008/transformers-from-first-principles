"""Figures for Chapter 30, The KV Cache, Cost and Speed.  Numbers are read from 04-code/ch30/outputs/ (run the chapter's
scripts first: 04-code/ch30/RUN.txt).  Run from the project root: python3 03-figures/src/ch30_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch30"); os.makedirs(OUT, exist_ok=True)
J = lambda n: json.load(open(os.path.join(ROOT, "04-code", "ch30", "outputs", n + ".json")))
p = lambda n: os.path.join(OUT, n)
A, M = J("cache_gpt2"), J("cache_more")

# 01 Signature 9: the KV cache, what is recomputed and what is kept
fig, ax = canvas(12, 6)
title(ax, "Signature 9. The cache after the running sentence, and the seventh token", y=59)
words = ["the", " cat", " sat", " on", " the", " mat"]
for i, w in enumerate(words):
    x = 6 + i * 13
    box(ax, x, 6, 11, 7, repr(w).strip("'"), color=AMBER, fs=10)
    box(ax, x, 22, 11, 7, f"k{i + 1}", color=SLATEBLUE, fs=10)
    box(ax, x, 33, 11, 7, f"v{i + 1}", color=SLATEBLUE, fs=10)
    arrow(ax, x + 5.5, 13, x + 5.5, 22)
x = 6 + 6 * 13
box(ax, x, 6, 11, 7, "new token", color=AMBER, fs=10)
box(ax, x, 22, 11, 7, "k7", color=CORAL, fs=10); box(ax, x, 33, 11, 7, "v7", color=CORAL, fs=10)
box(ax, x, 45, 11, 7, "q7", color=BLUE, fs=10)
arrow(ax, x + 5.5, 13, x + 5.5, 22)
ax.add_patch(plt.Rectangle((4.5, 20.5), 78, 21, fill=False, ec=SLATEBLUE, lw=2, ls="--"))
label(ax, 43, 44, "kept from earlier passes: 12 layers x 12 heads x 6 positions x 64 x 2 = 110,592 numbers (442,368 bytes)", color=SLATEBLUE, fs=9)
label(ax, 43, 3, "computed once per token, never again", color=GREY, fs=9)
label(ax, x + 13, 40, "only the new\ntoken's query,\nkey and value\nare computed", color=CORAL, fs=9, ha="left")
save(fig, p("fig30-01-signature9.png"))

# 02 seconds for 32 new tokens, cached and uncached
C = A["C"]; P = sorted(int(k) for k in C)
fig, ax = plt.subplots(figsize=(8.4, 4.8))
ax.plot(P, [C[str(k)]["uncached"] for k in P], "o-", color=VERM, label="no cache: a full pass per token")
ax.plot(P, [C[str(k)]["cached"] for k in P], "s-", color=SLATEBLUE, label="cache: one prefill, then one-token passes")
for k in P:
    ax.text(k, C[str(k)]["uncached"] * 1.15, f"{C[str(k)]['uncached']:.1f} s", ha="center", fontsize=9, color=VERM)
    ax.text(k, C[str(k)]["cached"] * 0.72, f"{C[str(k)]['cached']:.1f} s", ha="center", fontsize=9, color=SLATEBLUE)
ax.set_yscale("log"); ax.set_xlabel("prompt length (GPT-2 tokens)"); ax.set_ylabel("seconds for 32 new tokens (log scale)")
ax.legend(fontsize=9, loc="upper left"); ax.set_title("GPT-2 small, one CPU thread, 32-bit, greedy")
save(fig, p("fig30-02-timing.png"))

# 03 the cost of compressing the cache
G = M["G"]
fig, ax = plt.subplots(figsize=(9.2, 5.0))
mk = {"bit": ("o", BLUE), "keep": ("s", PLUM)}
for lab, v in G.items():
    m, col = mk["keep"] if lab.startswith("keep") else mk["bit"]
    if lab.startswith("32"): m, col = "*", NAVY
    ax.scatter(v["ratio"], v["bpc"], marker=m, s=70, color=col, zorder=3)
    dy = -12 if "keys per channel" in lab and lab.startswith("4") else 2
    ax.annotate(lab.replace(" keys and values", "").replace(" bits per character", ""), (v["ratio"], v["bpc"]), xytext=(6, dy), textcoords="offset points", fontsize=8)
ax.axhline(G["32-bit (no compression)"]["bpc"], color=NAVY, ls=":", lw=1)
ax.set_xscale("log", base=2); ax.set_xlim(0.8, 40)
ax.set_xlabel("times smaller than the 32-bit cache at 1,024 tokens (log scale)"); ax.set_ylabel("bits per character (held-out Carroll)")
ax.set_title("Rounding (circles) and evicting (squares) GPT-2's cache")
save(fig, p("fig30-03-compress.png"))
print("ok", sorted(os.listdir(OUT)))
