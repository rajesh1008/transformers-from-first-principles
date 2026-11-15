"""Figures for Chapter 29, Loading the Real Thing.  Numbers are parsed from 04-code/ch29/outputs/*.txt (run the chapter's
scripts first: 04-code/ch29/RUN.txt).  Run from the project root: python3 03-figures/src/ch29_figs.py"""
import os, sys, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch29"); os.makedirs(OUT, exist_ok=True)
T = lambda n: open(os.path.join(ROOT, "04-code", "ch29", "outputs", n + ".txt")).read()
p = lambda n: os.path.join(OUT, n)

# 01 Signature 8: the real thing, Tiny's stack with the released models' numbers written on it
cols = [("Tiny", "2,766,592 learned", "12 layers, 4 heads x 32", "width 128, FF 512 (GELU)", "learned positions, trained 128", "tied output, vocab 2,048"),
        ("GPT-2 small (2019)", "124,439,808 learned", "12 layers, 12 heads x 64", "width 768, FF 3,072 (GELU tanh)", "learned positions, 1,024", "tied output, vocab 50,257"),
        ("SmolLM2-135M (2024)", "134,515,008 learned", "30 layers, 9 query / 3 KV heads", "width 576, SwiGLU 1,536", "rotary, theta 100,000", "tied output, vocab 49,152")]
fig, ax = canvas(11, 7)
title(ax, "Signature 8. One design, three sets of numbers", y=69)
for i, c in enumerate(cols):
    x = 4 + i * 36
    box(ax, x, 54, 32, 8, c[0], color=NAVY, fs=11.5, sub=c[1])
    box(ax, x + 1, 41, 30, 7, "final norm, " + c[5], color=TEAL, fs=8.3, weight="normal")
    box(ax, x + 3, 20, 26, 14, "block x L", color=BLUE, fs=10.5, sub=c[2])
    label(ax, x + 16, 17.5, c[3], fs=9)
    box(ax, x + 3, 6, 26, 8, "token + position", color=AMBER, fs=10, sub=c[4])
    arrow(ax, x + 16, 14, x + 16, 16.2); arrow(ax, x + 16, 34, x + 16, 41); arrow(ax, x + 16, 48, x + 16, 54)
label(ax, 55, 2, "Changes from Tiny: biases on every map and GELU tanh (GPT-2); RMSNorm, rotary positions, shared key-value heads, SwiGLU (SmolLM2)", color=CORAL, fs=9)
save(fig, p("fig29-01-signature8.png"))

# 02 how far each loading mistake moves the log-probabilities
L, G = T("load_gpt2"), T("smollm2")
rows = [(m.group(1).strip(), float(m.group(2))) for m in re.finditer(r"^  (\S.{0,46}?)\s+([0-9.]+e[+-]\d+)\s+same top", L + "\n" + G, re.M)]
rows = [(a, b) for a, b in rows]
fig, ax = plt.subplots(figsize=(9.6, 5.2))
y = np.arange(len(rows))
ax.barh(y, [b for _, b in rows], color=[CORAL if b > 1 else AMBER for _, b in rows])
ax.set_yticks(y); ax.set_yticklabels([a for a, _ in rows], fontsize=9); ax.invert_yaxis()
ax.set_xscale("log"); ax.axvline(3.242e-05, color=TEAL, ls="--"); ax.axvline(1e-4, color=NAVY, ls=":")
ax.text(3.4e-05, len(rows) - 0.4, "correct port 3.2e-5", color=TEAL, fontsize=9); ax.text(1.1e-4, len(rows) - 1.1, "bar 1e-4", color=NAVY, fontsize=9)
ax.set_xlim(1e-5, 1e4)
ax.set_xlabel("largest |difference| in log-probabilities against the reference (nats, log scale)")
ax.set_title("Nine loading mistakes on GPT-2 (top five) and SmolLM2 (bottom four)")
save(fig, p("fig29-02-breakages.png"))

# 03 bits per character, three models, two window settings
labs = ["Tiny", "GPT-2 small", "SmolLM2-135M"]
short, long_ = [1.981, 1.876, 1.301], [2.439, 1.564, 1.119]
assert "1.9812" in T("tiny_bpc") and "1.8764" in T("gpt2_short") and "1.3006" in G and "1.5640" in T("gpt2_decode") and "1.1194" in G and "2.4392" in T("tiny_bpc")
x = np.arange(3); w = 0.38
fig, ax = plt.subplots(figsize=(8.4, 4.8))
b1 = ax.bar(x - w / 2, short, w, color=BLUE, label="windows of about 350 characters")
b2 = ax.bar(x + w / 2, long_, w, color=AMBER, label="windows of 1,024 tokens")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03, f"{b.get_height():.3f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("bits per character (lower is better)"); ax.set_ylim(0, 2.8)
ax.legend(fontsize=9, loc="upper right")
ax.set_title("Held-out Carroll, 10,919 characters, three tokenizers")
save(fig, p("fig29-03-bpc.png"))

# 04 chat formats on SmolLM2-135M-Instruct
E = T("tokenizers_templates")
fm = [(m.group(1).strip(), int(m.group(2)), [int(v) for v in m.group(3).split(",")]) for m in
      re.finditer(r"^  (\S.{0,35}?)\s+ended with <\|im_end\|> within 40 tokens: (\d) of 5; tokens written \[([^\]]*)\]", E, re.M)]
fig, ax = plt.subplots(figsize=(9.6, 4.6))
for i, (lab, s, lens) in enumerate(fm):
    ax.scatter(lens, [i] * len(lens), s=60, color=[TEAL if v < 40 else CORAL for v in lens], zorder=3)
ax.set_yticks(range(len(fm))); ax.set_yticklabels([f[0] for f in fm], fontsize=9.5); ax.invert_yaxis()
ax.set_xlabel("tokens written before <|im_end|> (40 = never ended; 1 = ended at once)"); ax.set_xlim(-1, 43)
ax.set_title("SmolLM2-135M-Instruct, five questions, four prompt formats, greedy")
save(fig, p("fig29-04-templates.png"))
print("ok", sorted(os.listdir(OUT)), len(rows), len(fm))
