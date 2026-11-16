"""Figures for Chapter 13, Normalisation and Residuals.  Every number is read from 04-code/ch13/outputs or
recomputed here from tiny/norm.py and the hand tables.  Run from the project root: python3 03-figures/src/ch13_figs.py"""
import os, sys, re, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code")); sys.path.insert(0, os.path.join(ROOT, "04-code", "ch13"))
from style import *
import matplotlib.pyplot as plt
from tiny.norm import layer_norm, rms_norm
setup()
OUT = os.path.join(ROOT, "03-figures", "ch13"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch13", "outputs")

# 01 the Interlude's list before and after normalising
L = np.array([2, 4, 4, 4, 5, 5, 7, 9.0]); Z = layer_norm(L)
fig, axs = plt.subplots(2, 1, figsize=(8.4, 3.6))
for ax, v, c, t, ticks in ((axs[0], L, AMBER, "the list: average 5, spread 2", range(0, 11)), (axs[1], Z, NAVY, "normalised: average 0, spread 1", np.arange(-2, 2.6, 0.5))):
    vals, counts = np.unique(np.round(v, 3), return_counts=True)
    for x_, n_ in zip(vals, counts):
        for k in range(n_): ax.plot(x_, 0.25 + 0.3 * k, "o", color=c, ms=10)
    m = v.mean(); s = v.std()
    ax.axvline(m, color=PLUM, lw=1.6); ax.axvspan(m - s, m + s, color=TINT[PLUM], zorder=0)
    ax.set_yticks([]); ax.set_ylim(0, 1.4); ax.set_xticks(list(ticks)); ax.set_title(t, fontsize=11.5); ax.grid(False)
axs[0].text(5.1, 1.15, "average", color=PLUM, fontsize=9.5); axs[0].text(7.05, 1.15, "average + spread", color=PLUM, fontsize=9.5)
fig.tight_layout(); save(fig, os.path.join(OUT, "fig13-01-interlude.png"))

# 02 layer norm and RMSNorm on the toy rows (two numbers per row)
sys.path.insert(0, os.path.join(ROOT, "04-code", "ch13"))
os.chdir(os.path.join(ROOT, "04-code", "ch13"))
from _common import hand_layer, X, WORDS
_, Y = hand_layer(X)
fig, ax = plt.subplots(figsize=(6.6, 6.0))
t = np.linspace(0, 2 * np.pi, 300); ax.plot(np.sqrt(2) * np.cos(t), np.sqrt(2) * np.sin(t), color=TEAL, lw=1.2, ls="--", label="every RMSNorm output lies on this circle")
for w, y in zip(WORDS, Y):
    r = rms_norm(y)
    ax.plot(*y, "o", color=AMBER, ms=8); ax.annotate(w, y, (y[0] + 0.08, y[1] + 0.08), fontsize=10, color=SLATE)
    ax.plot(*r, "s", color=TEAL, ms=7)
    ax.plot([0, y[0]], [0, y[1]], color=AMBER, lw=0.8, alpha=0.6)
ax.plot([1, -1], [-1, 1], "D", color=VERM, ms=10, label="the only two layer-norm outputs: (1, -1) and (-1, 1)")
ax.plot([], [], "o", color=AMBER, label="rows after Chapter 12's hand layer"); ax.plot([], [], "s", color=TEAL, label="the same rows after RMSNorm")
ax.set_aspect("equal"); ax.set_xlim(-1.8, 4.6); ax.set_ylim(-1.8, 4.6); ax.axhline(0, color="#999", lw=0.8); ax.axvline(0, color="#999", lw=0.8)
ax.legend(loc="upper left", fontsize=8.8); ax.set_title("Two numbers per row: layer norm keeps only which is bigger")
save(fig, os.path.join(OUT, "fig13-02-two-norms.png"))

# 03 four wirings
fig, ax = canvas(10, 5.2)
cols = [("no residual", "none"), ("residual only", "residual"), ("post-norm (2017, BERT)", "post"), ("pre-norm (GPT-2 and since)", "pre")]
for i, (t_, w) in enumerate(cols):
    x0 = 3 + i * 24.5
    label(ax, x0 + 9, 49, t_, color=NAVY, fs=10.5, weight="bold")
    box(ax, x0 + 2, 3, 14, 6, "row in", color=AMBER, fs=9.5)
    box(ax, x0 + 2, 41, 14, 5, "row out", color=AMBER, fs=9.5)
    if w == "pre":
        box(ax, x0 + 9, 13, 10, 5, "norm", color=TEAL, fs=9.5); box(ax, x0 + 9, 22, 10, 7, "block", color=NAVY, fs=9.5)
        arrow(ax, x0 + 9, 9, x0 + 14, 13); arrow(ax, x0 + 14, 18, x0 + 14, 22); arrow(ax, x0 + 14, 29, x0 + 10, 35)
    else:
        box(ax, x0 + 4, 18, 10, 7, "block", color=NAVY, fs=9.5); arrow(ax, x0 + 9, 9, x0 + 9, 18)
        arrow(ax, x0 + 9, 25, x0 + 9, 33 if w != "none" else 41)
    if w != "none":
        ax.add_patch(Circle((x0 + 9, 35), 1.6, fc="white", ec=SLATE, lw=1.4)); label(ax, x0 + 9, 35, "+", fs=12, weight="bold")
        arrow(ax, x0 + 3, 9, x0 + 7.6, 35, color=PLUM, rad=-0.0, lw=2.4) if w != "pre" else arrow(ax, x0 + 4, 9, x0 + 7.6, 35, color=PLUM, lw=2.4)
        label(ax, x0 + 1.5, 24, "kept", color=PLUM, fs=9, rotation=80)
        if w == "post":
            box(ax, x0 + 4, 37.2, 10, 3.3, "norm", color=TEAL, fs=8.5)
            arrow(ax, x0 + 9, 36.6, x0 + 9, 37.2)
        else:
            arrow(ax, x0 + 9, 36.6, x0 + 9, 41)
save(fig, os.path.join(OUT, "fig13-03-wirings.png"))

# 04-06 the depth experiment
P = json.load(open(os.path.join(CH, "placement.json")))
DEPTHS = [1, 2, 4, 6, 12, 24, 48]
names = {"none": ("no residual", GREY, ":"), "residual": ("residual only", VERM, "-."), "post": ("post-norm", PLUM, "--"), "pre": ("pre-norm", BLUE, "-"), "keel": ("post-norm, kept row weighted (Jan 2026)", TEAL, "-")}
fig, ax = plt.subplots(figsize=(8.4, 4.6))
for w, (lab, c, ls) in names.items():
    ax.plot(DEPTHS, [P[f"0.088/{w}"][str(d)][0] for d in DEPTHS], marker="o", color=c, ls=ls, label=lab)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(DEPTHS, DEPTHS); ax.set_xlabel("layers in the stack"); ax.set_ylabel("mean row length at the top")
ax.legend(fontsize=9, loc="upper left"); ax.set_title("Row length with depth: untrained Tiny layers, tables of spread 0.088")
save(fig, os.path.join(OUT, "fig13-04-length.png"))

fig, ax = plt.subplots(figsize=(8.4, 4.6))
for w in ("post", "pre", "keel", "residual"):
    lab, c, ls = names[w]; ax.plot(DEPTHS, [P[f"0.088/{w}"][str(d)][1] for d in DEPTHS], marker="o", color=c, ls=ls, label=lab)
dd = np.array(DEPTHS); ax.plot(dd, 0.611 * 2.0 ** (-(2 * dd - 2) / 2), color=PLUM, lw=1, ls=":", label="post-norm guide: halves in squared size per addition")
ax.set_xscale("log", base=2); ax.set_xticks(DEPTHS, DEPTHS); ax.set_ylim(-0.02, 1.05); ax.set_xlabel("layers in the stack"); ax.set_ylabel("share of the top row that is the input row, untouched")
ax.legend(fontsize=9); ax.set_title("How much of the input row is still carried at the top")
save(fig, os.path.join(OUT, "fig13-05-carried.png"))

fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, sp in zip(axs, ("0.020", "0.088")):
    for w, (lab, c, ls) in names.items():
        ax.plot(DEPTHS, [P[f"{sp}/{w}"][str(d)][2] for d in DEPTHS], marker="o", color=c, ls=ls, label=lab)
    ax.set_xscale("log", base=2); ax.set_xticks(DEPTHS, DEPTHS); ax.set_xlabel("layers"); ax.set_title(f"tables of spread {sp}", fontsize=11.5)
axs[0].set_ylabel("token similarity at the top (1 = all rows the same)"); axs[1].legend(fontsize=8.5, loc="lower right")
fig.suptitle("Do the rows stay different from each other?", x=0.06, ha="left", fontsize=13, color=NAVY, weight="bold")
fig.tight_layout(); save(fig, os.path.join(OUT, "fig13-06-similarity.png"))

# 07 GPT-2's residual stream
S = np.load(os.path.join(CH, "gpt2_stream_lengths.npy"))
fig, ax = plt.subplots(figsize=(8.4, 4.4)); xx = np.arange(len(S)); labs = [str(i) for i in range(12)] + ["top"]
ax.plot(xx, S[:, 0], marker="o", color=VERM, label="seat 0 (the end-of-text token)")
ax.plot(xx, np.median(S[:, 1:], axis=1), marker="o", color=BLUE, label="median of seats 1 to 1,023")
ax.fill_between(xx, np.percentile(S[:, 1:], 10, axis=1), np.percentile(S[:, 1:], 90, axis=1), color=TINT[BLUE], label="10th to 90th percentile")
ax.set_yscale("log"); ax.set_xticks(xx, labs); ax.set_xlabel("entering layer"); ax.set_ylabel("row length"); ax.legend(fontsize=9, loc="center right")
ax.set_title("GPT-2 small's residual stream on 1,024 tokens of the novel")
save(fig, os.path.join(OUT, "fig13-07-gpt2-stream.png"))

# 08 GPT-2 interventions
R = json.load(open(os.path.join(CH, "gpt2_norms.json"))); b = R["baseline"]["loss"]
items = [("all 25 frozen,\nseat 0 keeps its own", "freeze_all_keep0"), ("only the\ntwelve ln_1", "freeze_ln1"), ("only the\ntwelve ln_2", "freeze_ln2"), ("only ln_f", "freeze_lnf"), ("RMSNorm\neverywhere", "rms")]
fig, ax = plt.subplots(figsize=(8.4, 4.2)); v = [R[k]["loss"] - b for _, k in items]
ax.bar(range(len(items)), v, color=[TINT[TEAL]] * 4 + [TINT[PLUM]], edgecolor=[TEAL] * 4 + [PLUM])
for i, val in enumerate(v): ax.text(i, val + 0.005, f"+{val:.3f}", ha="center", fontsize=10, color=SLATE)
ax.set_xticks(range(len(items)), [t for t, _ in items], fontsize=9.5); ax.set_ylabel("change in loss"); ax.set_ylim(0, 0.25)
ax.text(-0.4, 0.235, "freezing seat 0's divisor too: the model breaks (infinite loss)", fontsize=9.5, color=VERM)
ax.set_title(f"GPT-2 small with its norms frozen or simplified (baseline {b:.3f})")
save(fig, os.path.join(OUT, "fig13-08-gpt2-interventions.png"))

# 09 Lab breakage 1
txt = open(os.path.join(CH, "lab13.txt")).read()
blk = re.findall(r"^\s+(correct \(pre-norm\)|no residual, pre-norm|no residual, no norm)\s+length\s+(.*)\n\s+similar\s*(.*)$", txt, re.M)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9)); ly = [1, 2, 3, 6, 9, 12]
for (nm, ls_, ss), c in zip(blk, (BLUE, VERM, GREY)):
    lv = [float(t) for t in ls_.split()]; sv = [float(t) for t in ss.split()]
    axs[0].plot(ly, [max(t, 1e-40) for t in lv], marker="o", color=c, label=nm); axs[1].plot(ly, sv, marker="o", color=c, label=nm)
axs[0].set_yscale("log"); axs[0].set_xlabel("layer"); axs[0].set_ylabel("mean row length"); axs[1].set_xlabel("layer"); axs[1].set_ylabel("token similarity")
axs[0].legend(fontsize=8.5); fig.suptitle("Lab 13, Breakage 1: delete the residual additions", x=0.06, ha="left", fontsize=13, color=NAVY, weight="bold")
fig.tight_layout(); save(fig, os.path.join(OUT, "fig13-09-lab-breakage.png"))
print("9 figures in", OUT)
