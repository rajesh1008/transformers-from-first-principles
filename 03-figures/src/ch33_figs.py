"""Figures for Chapter 33, Quantisation and Running Tiny Small.  Numbers are read from 04-code/ch33/outputs/ (run
04-code/ch33/RUN.txt first).  Run from the project root: python3 03-figures/src/ch33_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch33"); os.makedirs(OUT, exist_ok=True)
J = lambda n: json.load(open(os.path.join(ROOT, "04-code", "ch33", "outputs", n + ".json")))
p = lambda n: os.path.join(OUT, n)
G, Q, T = J("quant_gpt2"), J("quant_gptq"), J("tiny_quant")
base = G["B"]["32"]

# 01 GPT-2: bits per character against bits per weight, four ways of sharing a scale
fig, ax = plt.subplots(figsize=(7.6, 5.6))
bits = [8, 6, 4, 3, 2]
for key, col, mk, lab in (("tensor", GREY, "s", "one scale per table"), (None, BLUE, "o", "per output channel"),
                          (128, TEAL, "^", "per 128 inputs"), (32, PLUM, "D", "per 32 inputs")):
    bs = [b for b in bits if f"{b} {key}" in G["B"]]
    ax.plot(bs, [G["B"][f"{b} {key}"] for b in bs], marker=mk, color=col, label=lab)
ax.axhline(base, color=VERM, ls="--", lw=1)
ax.annotate(f"32-bit: {base:.3f}", (2.05, base), xytext=(0, -15), textcoords="offset points", color=VERM, fontsize=10)
ax.set_xlim(8.3, 1.7); ax.set_ylim(1.3, 4.6)
ax.set_xlabel("bits per weight (before scales)"); ax.set_ylabel("bits per character, held-out Carroll")
ax.legend(frameon=False, loc="upper left")
ax.set_title("GPT-2 small, 48 linear maps rounded to nearest", fontsize=12)
save(fig, p("fig33-01-grid.png"))

# 02 what breaks first at 3 bits per output channel: by kind of map, then by layer
D = G["D"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.8), gridspec_kw={"width_ratios": [1, 2.2]})
kinds = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
a1.bar(range(4), [D[k] - base for k in kinds], color=[BLUE, TEAL, VERM, PLUM])
a1.set_xticks(range(4)); a1.set_xticklabels(["attn\nqkv", "attn\nout", "mlp\nup", "mlp\ndown"], fontsize=9)
a1.set_ylabel("added bits per character"); a1.set_title("by kind (all 12 layers)", fontsize=11)
a2.bar(range(12), [D[f"layer {i}"] - base for i in range(12)], color=NAVY)
a2.set_xticks(range(12)); a2.set_xlabel("layer rounded (the other eleven stay 32-bit)"); a2.set_title("by layer", fontsize=11)
fig.suptitle("3-bit per output channel, one part at a time", x=0.02, ha="left", fontsize=12, color=NAVY, fontweight="bold")
save(fig, p("fig33-02-fragile.png"))

# 03 token table: the four 4-bit variants
labs = ["table 4-bit,\ninput lookup only", "table 4-bit,\noutput only", "output, final-norm\ntop 2 dims kept", "output, final-norm\ntop 8 dims kept", "table 8-bit\nper row"]
vals = [0.0059, 2.0527, 0.7562, 0.0568, 0.0252]
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.bar(range(5), vals, color=[TEAL, VERM, AMBER, AMBER, BLUE])
for i, v in enumerate(vals):
    ax.annotate(f"+{v:.3f}", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10, color=SLATE)
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=8.5)
ax.set_ylabel("added bits per character")
ax.set_title("The token table is two jobs; the output job breaks at 4 bits", fontsize=12)
save(fig, p("fig33-03-table.png"))

# 04 GPTQ against round-to-nearest, and Tiny against GPT-2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.8))
labs = ["4-bit\nper 128", "3-bit\nper 128", "3-bit per\nchannel"]
rtn = [G["B"]["4 128"], G["B"]["3 128"], G["B"]["3 None"]]; gp = [Q["F"]["4 128"], Q["F"]["3 128"], Q["F"]["3 None"]]
x = np.arange(3)
a1.bar(x - 0.2, [r - base for r in rtn], 0.4, color=BLUE, label="round-to-nearest")
a1.bar(x + 0.2, [g - base for g in gp], 0.4, color=PLUM, label="GPTQ")
a1.set_xticks(x); a1.set_xticklabels(labs, fontsize=9); a1.set_ylabel("added bits per character"); a1.legend(frameon=False)
a1.set_title("GPT-2, calibrated on 16 x 256 tokens", fontsize=11)
bt = [8, 4, 3, 2]
tg = [G["B"][f"{b} None"] - base for b in bt]
tt = [T["I"][f"{b} None False"]["bpc"] - T["I"]["32"] for b in bt]
a2.plot(bt, tg, "o-", color=BLUE, label="GPT-2 small (124M)")
a2.plot(bt, tt, "s-", color=VERM, label="Tiny (2.8M)")
a2.set_xlim(8.3, 1.7); a2.set_xticks(bt); a2.set_xlabel("bits per weight, per output channel, maps only")
a2.set_ylabel("added bits per character"); a2.legend(frameon=False); a2.set_title("the small model rounds better", fontsize=11)
save(fig, p("fig33-04-gptq-tiny.png"))
