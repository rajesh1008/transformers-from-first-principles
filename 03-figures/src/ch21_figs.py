"""Figures for Chapter 21, Mixed Precision, Memory and Where the Training Budget Goes.  Every measured number is
read from 04-code/ch21/outputs/*.json; run the chapter's scripts first, in the order of 04-code/ch21/RUN.txt.
fig21-01 draws the bit layouts of the formats (no measurement).  Run from the project root:
python3 03-figures/src/ch21_figs.py"""
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code"))
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch21"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch21", "outputs")
js = lambda n: json.load(open(os.path.join(CH, n + ".json")))
p = lambda n: os.path.join(OUT, n)
NU, MB, BU = js("nudges"), js("memory_budget"), js("budget")
TM = {os.path.basename(f)[12:-5]: json.load(open(f)) for f in glob.glob(os.path.join(CH, "train_mixed_*.json"))}

# 01 bit layouts
fig, ax = plt.subplots(figsize=(8.6, 4.4))
F = [("FP32", 8, 23), ("BF16", 8, 7), ("FP16", 5, 10), ("FP8 E5M2", 5, 2), ("FP8 E4M3", 4, 3), ("FP4 E2M1", 2, 1)]
for i, (n, e, m) in enumerate(F):
    y = len(F) - 1 - i; x = 0
    for w, col, lab in ((1, VERM, "sign"), (e, BLUE, "exponent"), (m, TEAL, "mantissa")):
        ax.barh(y, w, left=x, color=col, edgecolor="white", height=0.7)
        if w >= 2 or lab == "sign":
            ax.text(x + w / 2, y, str(w), ha="center", va="center", color="white", fontsize=9, weight="bold")
        x += w
    ax.text(-0.6, y, n, ha="right", va="center", fontsize=10, color=NAVY)
ax.set_xlim(-7, 33); ax.set_yticks([]); ax.set_xlabel("bits")
for col, lab, xx in ((VERM, "sign", 10), (BLUE, "exponent bits: range", 15), (TEAL, "mantissa bits: precision", 24)):
    ax.barh(-1.1, 1.2, left=xx, color=col, height=0.4); ax.text(xx + 1.5, -1.1, lab, va="center", fontsize=9)
ax.set_ylim(-1.6, len(F) - 0.4)
ax.set_title("Six number formats: exponent bits set the range, mantissa bits the precision")
for s in ("left", "right", "top"):
    ax.spines[s].set_visible(False)
save(fig, p("fig21-01-bits.png"))

# 02 where Tiny's nudges fall on the FP16 line
fig, ax = plt.subplots(figsize=(8.6, 4.4))
e = np.array(NU["edges"]); h = np.array(NU["hist"])
ax.bar(2.0 ** e[:-1], h, width=np.diff(2.0 ** e), align="edge", color=AMBER, alpha=0.85, label="Tiny's nudges, first batch (non-zero)")
ax.set_xscale("log")
for x, lab, col in ((5.96e-8, "FP16 smallest (5.96e−8)", VERM), (6.10e-5, "FP16 smallest normal (6.1e−5)", PLUM), (65504, "FP16 largest", VERM)):
    ax.axvline(x, color=col, ls="--", lw=1.2); ax.text(x * 1.2, h.max() * 0.9, lab, color=col, fontsize=8.5, rotation=90, va="top")
ax.axvspan(1e-12, 5.96e-8, color=VERM, alpha=0.08); ax.axvspan(5.96e-8, 6.1e-5, color=PLUM, alpha=0.06)
ax.set_xlim(1e-11, 1e5); ax.set_xlabel("size of a nudge (log scale)"); ax.set_ylabel("count")
ax.set_title("A fifth of Tiny's nudges sit in FP16's thin subnormal range; a few fall below it")
ax.legend(fontsize=9, loc="upper left")
save(fig, p("fig21-03-nudges-fp16.png"))

# 03 loss scale
fig, ax = plt.subplots(figsize=(8.6, 4.2))
k = [r["k"] for r in NU["scale"]]
for key, col, lab in (("zero", VERM, "become exactly 0"), ("sub", PLUM, "subnormal (few digits left)"), ("inf", NAVY, "overflow to infinity")):
    ax.plot(k, [100 * r[key] for r in NU["scale"]], "-o", color=col, label=lab)
ax.set_yscale("symlog", linthresh=1e-4); ax.set_xticks(k); ax.set_xticklabels([f"2^{x}" for x in k])
ax.set_xlabel("loss scale"); ax.set_ylabel("per cent of Tiny's nudges, stored in FP16")
ax.set_title("Multiplying the loss by 2^16 moves every nudge into FP16's range; 2^24 is too far")
ax.legend(fontsize=9)
save(fig, p("fig21-04-loss-scale.png"))

# 04 lost updates
fig, ax = plt.subplots(figsize=(8.6, 4.2))
L = NU["lost"]; st = [r["step"] for r in L]
ax.plot(st, [100 * r["tables"] for r in L], "-o", color=BLUE, label="two-axis tables (spread 0.02)")
ax.plot(st, [100 * r["gains"] for r in L], "-s", color=VERM, label="norm gains (1.0)")
ax.plot(st, [100 * r["biases"] for r in L], "-^", color=TEAL, label="biases (0)")
ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlabel("size of the step (log scale, smaller to the right)")
ax.set_ylabel("per cent of steps a BF16 table cannot record")
ax.set_title("A BF16 table drops every step smaller than half the gap to its next number"); ax.legend(fontsize=9)
save(fig, p("fig21-02-lost-updates.png"))

# 05 seven recipes at lr 1e-3
fig, ax = plt.subplots(figsize=(8.6, 4.6))
for key, col, lab in (("fp32", NAVY, "32-bit"), ("bf16", BLUE, "mixed BF16"), ("fp16", AMBER, "mixed FP16, no loss scale"),
                      ("fp16_scaled", TEAL, "mixed FP16, loss scaled"), ("bf16_state", PLUM, "mixed BF16, Adam state BF16"),
                      ("w16_nearest", VERM, "BF16 tables only, nearest"), ("w16_sr", GREEN, "BF16 tables only, stochastic")):
    c = np.array(TM[key]["curve"]); ax.plot(c[:, 0], c[:, 2], "-", color=col, lw=1.6, label=f"{lab}: {TM[key]['held_end']:.4f}")
ax.set_xlabel("step"); ax.set_ylabel("held-out loss (computed in 32-bit)"); ax.set_ylim(5.0, 7.8)
ax.set_title("Adam lr 10⁻³, warmup and cosine: seven recipes end within 0.05 of each other")
ax.legend(fontsize=8.5)
save(fig, p("fig21-05-recipes.png"))

# 06 low learning rate: the stall
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.2))
for key, col, lab in (("fp32_lr3e-5", NAVY, "32-bit"), ("bf16_lr3e-5", BLUE, "mixed BF16"),
                      ("w16_nearest_lr3e-5", VERM, "BF16 tables, nearest"), ("w16_sr_lr3e-5", GREEN, "BF16 tables, stochastic")):
    c = np.array(TM[key]["curve"]); a1.plot(c[:, 0], c[:, 2], "-", color=col, lw=1.8 if "nearest" in key else 1.3,
                                              ls="--" if key.startswith("bf16") else "-", label=f"{lab}: {TM[key]['held_end']:.4f}")
    if TM[key].get("lost"):
        a2.plot(100 * np.array(TM[key]["lost"]), color=col, lw=1.2, label=lab)
a1.set_xlabel("step"); a1.set_ylabel("held-out loss"); a1.set_title("Adam lr 3·10⁻⁵, constant"); a1.legend(fontsize=8)
a2.set_xlabel("step"); a2.set_ylabel("per cent of steps lost in rounding"); a2.set_title("Steps a BF16 table did not record"); a2.legend(fontsize=8)
save(fig, p("fig21-06-stall.png"))

# 07 memory of one step
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.4))
T = MB["tiny"]; names = list(T)
short = ["32-bit", "mixed BF16", "pure BF16"]
states = [44.3, 44.3, 22.1]
a1.bar(np.arange(3) - 0.2, [T[n]["full"] / 1e6 for n in names], 0.38, color=AMBER, label="activations kept")
a1.bar(np.arange(3) + 0.2, [T[n]["ckpt"] / 1e6 for n in names], 0.38, color=TEAL, label="activations, each layer recomputed")
a1.plot(np.arange(3), states, "D", color=PLUM, label="weights + nudges + Adam")
a1.set_xticks(range(3)); a1.set_xticklabels(short); a1.set_ylabel("MB"); a1.set_title("Tiny, 8 × 128 tokens (measured)"); a1.legend(fontsize=8)
P7 = MB["P7"]
parts = [("weights 16-bit", 2), ("nudges 16-bit", 2), ("master 32-bit", 4), ("Adam m, v", 8)]
bot = 0
for (lab, b), col in zip(parts, (NAVY, BLUE, SLATEBLUE, PLUM)):
    a2.bar(0, b * P7 / 1e9, bottom=bot, color=col, label=lab); bot += b * P7 / 1e9
a2.bar(1, MB["act_bnd"] / 1e9, color=AMBER, label="activations (per token)")
a2.bar(1, MB["act_att"] / 1e9, bottom=MB["act_bnd"] / 1e9, color=VERM, label="attention tables")
a2.bar(2, 32 * 4096 * 4096 * 2 / 1e9, color=TEAL, label="layer inputs only")
a2.set_xticks(range(3)); a2.set_xticklabels(["states", "activations\nkept", "each layer\nrecomputed"])
a2.set_ylabel("GB"); a2.set_title("7B model, 4,096 tokens (counted)"); a2.legend(fontsize=7.5)
save(fig, p("fig21-07-memory.png"))

# 08 time of one step
fig, ax = plt.subplots(figsize=(8.6, 4.4))
R = BU["rows"]; keys = list(R)
labs = ["32-bit", "mixed BF16", "pure BF16\n(no Adam)", "32-bit,\nrecomputed", "mixed BF16,\nrecomputed"]
f = [R[k][0] for k in keys]; b = [R[k][1] for k in keys]; o = [R[k][2] for k in keys]
ax.bar(range(5), f, color=BLUE, label="forward"); ax.bar(range(5), b, bottom=f, color=AMBER, label="backward")
ax.bar(range(5), o, bottom=np.add(f, b), color=PLUM, label="Adam")
for i in range(5):
    ax.text(i, f[i] + b[i] + o[i] + 5, f"{f[i] + b[i] + o[i]:.0f} ms", ha="center", fontsize=9)
ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=9); ax.set_ylabel("ms, one thread, fastest of five")
ax.set_title("One training step of Tiny on the book's CPU, 11 November 2026"); ax.legend(fontsize=9)
save(fig, p("fig21-08-step-time.png"))
print("ch21 figures written")
