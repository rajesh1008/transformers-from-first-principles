"""Figures for Chapter 16, Reading a Config File.  Every number is read from 04-code/ch16/outputs
(count.json, memory.json, cost.json) or recomputed with 04-code/ch16/reader.py from the configuration
files in 04-code/ch16/configs.  Run from the project root: python3 03-figures/src/ch16_figs.py"""
import os, sys, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code")); sys.path.insert(0, os.path.join(ROOT, "04-code", "ch16"))
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch16"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch16", "outputs")
js = lambda n: json.load(open(os.path.join(CH, n + ".json")))
from _common import load_config, load_header, MODELS
from reader import spec, tiny_spec, count, total
CNT, MEM, COST = js("count"), js("memory"), js("cost")
specs = [("Tiny", tiny_spec())] + [(n, spec(load_config(n))) for n in MODELS]

# 01 from a configuration to a count: the fields on the left, the parts they decide on the right
fig, ax = canvas(9.6, 6.2)
ax.set_xlim(0, 100); ax.set_ylim(0, 102)   # draw on a 100 x 100 grid
fields = [("vocab_size", AMBER), ("hidden_size (width)", NAVY), ("num_hidden_layers", NAVY),
          ("num_attention_heads, head_dim", NAVY), ("num_key_value_heads", SLATEBLUE),
          ("intermediate_size, hidden_act", NAVY), ("norm type, bias", TEAL),
          ("max_position_embeddings", AMBER), ("tie_word_embeddings", PLUM)]
parts = [("token table  V x d", AMBER, [0, 1]), ("attention  q, k, v, o", NAVY, [1, 3, 4, 6]),
         ("feed-forward  2 or 3 tables of d x d_ff", NAVY, [1, 5, 6]), ("norms  d or 2d each", TEAL, [1, 6]),
         ("x layers", NAVY, [2]), ("position table (learned only)", AMBER, [7, 1]),
         ("output table (only if untied)", PLUM, [8, 0, 1])]
fy = [92 - 10.2 * i for i in range(len(fields))]
py = [90 - 12.6 * i for i in range(len(parts))]
for (f, col), y in zip(fields, fy):
    box(ax, 2, y - 3.5, 34, 7, f, color=col, fs=9.5)
for (p, col, src), y in zip(parts, py):
    box(ax, 62, y - 4, 36, 8, p, color=col, fs=9.5)
    for s_ in src:
        arrow(ax, 36, fy[s_], 62, y, color=GREY, lw=0.9, ms=8)
label(ax, 50, 99, "Nine settings decide every table's shape", color=NAVY, fs=12, weight="bold")
save(fig, os.path.join(OUT, "fig16-01-config-to-parts.png"))

# 02 where the learned numbers are, four models, shares of each total
fig, ax = plt.subplots(figsize=(9.0, 4.6))
cats = [("token table", AMBER), ("position table", "#e0b860"), ("attention", NAVY), ("feed-forward", BLUE), ("norms", TEAL)]
for i, (name, s) in enumerate(specs):
    rows = {p: n for p, _, n in count(s)}
    L = s["layers"]; T = total(s)
    vals = [rows["token table"], rows.get("position table", 0), L * rows["attention, per layer"],
            L * rows["feed-forward, per layer"], L * rows["norms, per layer"] + rows["final norm"]]
    left = 0
    for (cn, col), v in zip(cats, vals):
        ax.barh(i, v / T * 100, left=left, color=col, edgecolor="white", label=cn if i == 0 else None)
        if v / T > 0.06:
            ax.text(left + v / T * 50, i, f"{v / T * 100:.1f}%", ha="center", va="center", color="white", fontsize=9, weight="bold")
        left += v / T * 100
    ax.text(101, i, f"{T:,}", va="center", fontsize=9)
ax.set_yticks(range(len(specs))); ax.set_yticklabels([n for n, _ in specs]); ax.invert_yaxis()
ax.set_xlim(0, 125); ax.set_xlabel("per cent of the model's learned numbers")
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.45, 1.16), fontsize=8.5, frameon=False)
save(fig, os.path.join(OUT, "fig16-02-where.png"))

# 03 what the weights file holds against what is learned
fig, ax = plt.subplots(figsize=(8.6, 4.0))
rows = [("GPT-2 small", CNT["GPT-2 small"]["hand"], CNT["GPT-2 small"]["file_numbers"] - CNT["GPT-2 small"]["hand"], "12 stored masks"),
        ("SmolLM2-135M", CNT["SmolLM2-135M"]["hand"], 0, "nothing extra"),
        ("Qwen3-0.6B", CNT["Qwen3-0.6B"]["hand"], CNT["Qwen3-0.6B"]["file_numbers"] - CNT["Qwen3-0.6B"]["hand"], "a stored copy of the tied table")]
for i, (n, a, b, why) in enumerate(rows):
    ax.barh(i, a / 1e6, color=NAVY, label="learned numbers (hand count)" if i == 0 else None)
    ax.barh(i, b / 1e6, left=a / 1e6, color=VERM, hatch="//", edgecolor="white", label="stored, not learned" if i == 0 else None)
    ax.text((a + b) / 1e6 + 8, i, f"{a / 1e6:.1f} M + {b / 1e6:.1f} M ({why})", va="center", fontsize=9)
ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.set_xlim(0, 1250); ax.set_xlabel("millions of numbers in the weights file"); ax.legend(loc="upper right", fontsize=9)
save(fig, os.path.join(OUT, "fig16-03-file-vs-count.png"))

# 04 weights against the key-value cache at full context, 16-bit, log scale
fig, ax = plt.subplots(figsize=(8.6, 4.4))
names = [n for n, _ in specs]; x = np.arange(len(names)); w = 0.38
wts = [MEM[n]["weights"]["16"] / 1e6 for n in names]; kv = [MEM[n]["kv_full_16bit"] / 1e6 for n in names]
ax.bar(x - w / 2, wts, w, color=NAVY, label="weights, 16-bit")
ax.bar(x + w / 2, kv, w, color=SLATEBLUE, hatch="..", edgecolor="white", label="key-value cache at full context, batch 1, 16-bit")
for i in range(len(names)):
    ax.text(x[i] - w / 2, wts[i] * 1.15, f"{wts[i]:,.1f}", ha="center", fontsize=8.5)
    ax.text(x[i] + w / 2, kv[i] * 1.15, f"{kv[i]:,.1f}", ha="center", fontsize=8.5)
ax.set_yscale("log"); ax.set_ylim(1, 20000); ax.set_ylabel("MB (log scale)")
ax.set_xticks(x); ax.set_xticklabels([f"{n}\ncontext {s['context']:,}" for n, s in specs], fontsize=9)
ax.legend(fontsize=8.5, loc="upper left")
save(fig, os.path.join(OUT, "fig16-04-weights-vs-cache.png"))

# 05 Tiny's forward pass: hand FLOPs split, and the measured time
t = COST["tiny_1024"]; md = t["madds"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw=dict(width_ratios=[1.4, 1]))
ks = list(md); vals = [2 * md[k] / 1e9 for k in ks]
a1.barh(range(len(ks)), vals, color=[NAVY, VERM, VERM, BLUE, PLUM])
for i, v in enumerate(vals):
    a1.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)
a1.set_yticks(range(len(ks))); a1.set_yticklabels(ks, fontsize=8.5); a1.invert_yaxis()
a1.set_xlabel("billion FLOPs, one pass over 1,024 tokens"); a1.set_xlim(0, 4.2)
labels = ["hand count at the\ntable product's rate", "measured\nTiny forward"]
ms = [t["flops"] / t["rate_mm"] * 1e3, t["t_tiny_s"] * 1e3]
a2.bar([0, 1], ms, color=[TEAL, AMBER])
for i, v in enumerate(ms):
    a2.text(i, v + 20, f"{v:,.0f} ms", ha="center", fontsize=9)
a2.set_xticks([0, 1]); a2.set_xticklabels(labels, fontsize=8.5); a2.set_ylabel("milliseconds, one thread")
save(fig, os.path.join(OUT, "fig16-05-tiny-flops-time.png"))

# 06 three misreadings, each against the right figure
fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.8))
q = spec(load_config("Qwen3-0.6B")); sm = spec(load_config("SmolLM2-135M"))
from reader import kv_numbers_per_token as kvn
cases = [("1. tie ignored\nQwen3-0.6B, learned numbers (M)", total(q) / 1e6, total(dict(q, tied=False)) / 1e6),
         ("2. head width 64, not 128\nQwen3-0.6B, cache per token (numbers)", kvn(q), kvn(dict(q, head_dim=64))),
         ("3. key-value heads overlooked\nSmolLM2-135M, cache at 8,192 (MB)", kvn(sm) * 2 * 8192 / 1e6, kvn(dict(sm, kv_heads=9)) * 2 * 8192 / 1e6)]
for ax, (ttl, right, wrong) in zip(axs, cases):
    ax.bar([0, 1], [right, wrong], color=[NAVY, VERM])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["right", "misread"])
    for i, v in enumerate([right, wrong]):
        ax.text(i, v * 1.02, f"{v:,.1f}" if v < 10000 else f"{v:,.0f}", ha="center", fontsize=9)
    ax.set_title(ttl, fontsize=9); ax.set_ylim(0, max(right, wrong) * 1.18)
save(fig, os.path.join(OUT, "fig16-06-breakages.png"))
print("six figures written to", OUT)
