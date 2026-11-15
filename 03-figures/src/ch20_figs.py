"""Figures for Chapter 20, Optimizers and Learning-Rate Schedules.  Every measured number is read from
04-code/ch20/outputs/*.json (and Chapter 19's train_inits.json for fig20-07); run the chapter's scripts first,
in the order of 04-code/ch20/RUN.txt.  fig20-02 draws a synthetic stream (1 plus noise, seed 1865) to illustrate
running averages; it is labelled as such.  Run from the project root:  python3 03-figures/src/ch20_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code"))
from style import *
import matplotlib.pyplot as plt
import numpy as np
from tiny.optim import constant, warmup_cosine, wsd, linear_decay
setup()
OUT = os.path.join(ROOT, "03-figures", "ch20"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch20", "outputs")
js = lambda n, d=CH: json.load(open(os.path.join(d, n + ".json")))
p = lambda n: os.path.join(OUT, n)
V, LS = js("valley"), js("layer_steps")
TR = {}
for g in ("opt", "sched", "starts", "tune"):
    TR.update(js("train_" + g))
C19 = js("train_inits", os.path.join(ROOT, "04-code", "ch19", "outputs"))

# 01 the valley
fig, ax = plt.subplots(figsize=(8.6, 4.6))
xs, ys = np.meshgrid(np.linspace(-1.2, 1.3, 300), np.linspace(-1.2, 1.2, 300))
ax.contour(xs, ys, 0.5 * (xs ** 2 + 50 * ys ** 2), levels=[0.01, 0.05, 0.2, 0.5, 1, 2, 5, 10, 20], colors="#c3cad6", linewidths=0.8)
for key, col, name in (("sgd@0.039", VERM, "plain GD, lr 0.039"), ("momentum@0.04", TEAL, "momentum 0.9, lr 0.04"), ("adam@0.03", PLUM, "Adam, lr 0.03")):
    P = np.array(V[key]["path"]); n = V[key]["steps"]
    ax.plot(P[:, 0], P[:, 1], "-", color=col, lw=1.4, marker=".", ms=3, label=f"{name}: {n} steps")
ax.plot([1], [1], "o", color=NAVY); ax.text(1.03, 1.03, "start (1, 1)", color=NAVY, fontsize=9)
ax.set_xlabel("x (flat direction, curvature 1)"); ax.set_ylabel("y (steep, curvature 50)")
ax.set_title("A narrow valley: f = ½(x² + 50y²), steps until f < 10⁻⁴"); ax.legend(fontsize=9, loc="lower left")
ax.set_xlim(-0.6, 1.3); ax.set_ylim(-1.2, 1.2)
save(fig, p("fig20-01-valley.png"))

# 02 running averages (synthetic stream)
rng = np.random.default_rng(1865); x = 1 + 0.5 * rng.standard_normal(300)
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.plot(range(1, 301), x, color="#c3cad6", lw=0.8, label="the values: 1 plus noise (synthetic, seed 1865)")
for b, col in ((0.9, BLUE), (0.99, VERM)):
    a = 0.0; raw, cor = [], []
    for t, v in enumerate(x, 1):
        a = b * a + (1 - b) * v; raw.append(a); cor.append(a / (1 - b ** t))
    ax.plot(range(1, 301), raw, color=col, lw=2, ls="--", label=f"β = {b}, uncorrected")
    ax.plot(range(1, 301), cor, color=col, lw=2, label=f"β = {b}, bias-corrected")
ax.axvline(100, color=GREY, lw=0.8, ls=":"); ax.text(103, 0.15, "step 100: 1 − 0.99¹⁰⁰ = 0.63", color=GREY, fontsize=9)
ax.set_ylim(0, 2.2); ax.set_xlabel("step"); ax.set_ylabel("running average")
ax.set_title("A running average starts too low; dividing by 1 − βᵗ fixes it"); ax.legend(fontsize=8.5, loc="upper right", ncol=2)
save(fig, p("fig20-02-ema.png"))

# 03 layer steps
fig, ax = plt.subplots(figsize=(8.6, 4.4))
for key, col, ls in (("SGD lr 0.1 x1", VERM, "-"), ("SGD lr 0.1 x10", VERM, "--"), ("Adam lr 3e-3 x1", BLUE, "-"), ("Adam lr 3e-3 x10", BLUE, "--")):
    ax.plot(range(12), LS[key]["layers"], ls, color=col, marker="o", ms=4, label=key.replace(" x1", ", default start").replace(" x10", ", ×10 start").replace("0", "0", 1))
ax.set_yscale("log"); ax.set_xlabel("layer"); ax.set_ylabel("RMS of step 21's change to the layer's tables")
ax.set_title("Adam moves every layer about the same distance; SGD does not"); ax.legend(fontsize=9)
save(fig, p("fig20-03-layer-steps.png"))

# 04 schedules
fig, ax = plt.subplots(figsize=(8.6, 4.2))
S = range(400)
for f, col, name in ((constant(), GREY, "constant (no warmup, as in the runs)"), (warmup_cosine(40, 400, 0.1), BLUE, "warmup + cosine to 0.1"),
                     (linear_decay(40, 400), TEAL, "warmup + linear to 0"), (wsd(40, 400, 0.2), VERM, "warmup-stable-decay (last 20%)")):
    vals = [f(s) for s in S]
    ax.plot(S, vals, color=col, label=name)
ax.set_xlabel("step"); ax.set_ylabel("multiplier on lr"); ax.set_title("Four schedules over Tiny's 400 steps, 40 of warmup"); ax.legend(fontsize=9)
save(fig, p("fig20-04-schedules.png"))

def held(key):
    c = TR[key]["curve"]; return [s for s, _, _ in c], [h for _, _, h in c]
# 05 schedules run
fig, ax = plt.subplots(figsize=(8.6, 4.4))
for key, col, name in (("adam 3e-3 constant", GREY, "constant"), ("adam 3e-3 warmup+cosine", BLUE, "cosine"),
                       ("adam 3e-3 linear to 0", TEAL, "linear to 0"), ("adam 3e-3 wsd", VERM, "WSD")):
    s_, h_ = held(key); ax.plot(s_, h_, color=col, marker="o", ms=3, label=f"{name}: {TR[key]['held_end']:.4f}")
ax.axvline(320, color=VERM, lw=0.8, ls=":"); ax.text(323, 6.6, "WSD decay starts", color=VERM, fontsize=9)
ax.set_ylim(5.1, 7.8); ax.set_xlabel("step"); ax.set_ylabel("held-out loss (16 Looking-Glass windows)")
ax.set_title("Adam, lr 0.003: decaying at the end matters more than the shape"); ax.legend(fontsize=9)
save(fig, p("fig20-05-schedules-run.png"))

# 06 lr sweep
fig, ax = plt.subplots(figsize=(8.6, 4.4))
for keys, col, name in (([("adam 1e-3 constant", 1e-3), ("adam 3e-3 constant", 3e-3), ("adam 1e-2 constant", 1e-2)], BLUE, "Adam, constant"),
                        ([("adam 1e-3 warmup+cosine", 1e-3), ("adam 3e-3 warmup+cosine", 3e-3), ("adam 1e-2 warmup+cosine", 1e-2), ("adam 3e-2 warmup+cosine", 3e-2)], TEAL, "Adam, warmup + cosine"),
                        ([("muon 1e-3 warmup+cosine", 1e-3), ("muon 3e-3 warmup+cosine", 3e-3), ("muon 1e-2 warmup+cosine", 1e-2)], PLUM, "Muon, warmup + cosine")):
    ax.plot([l for _, l in keys], [TR[k]["held_end"] for k, _ in keys], "o-", color=col, label=name)
    for k, l in keys:
        ax.annotate(f"{TR[k]['held_end']:.2f}", (l, TR[k]["held_end"]), textcoords="offset points", xytext=(5, 4), fontsize=8, color=col)
ax.axhline(TR["sgd 0.1"]["held_end"], color=VERM, ls="--", lw=1); ax.text(1.05e-3, TR["sgd 0.1"]["held_end"] + 0.02, "plain GD, lr 0.1: 5.92", color=VERM, fontsize=9)
ax.axhline(TR["momentum 0.9, lr 0.06"]["held_end"], color=AMBER, ls="--", lw=1); ax.text(1.4e-3, TR["momentum 0.9, lr 0.06"]["held_end"] + 0.015, "momentum, lr 0.06: 5.26", color=AMBER, fontsize=9)
ax.set_xscale("log"); ax.set_xlabel("peak learning rate"); ax.set_ylabel("held-out loss after 400 steps")
ax.set_title("Learning-rate sweeps on Tiny"); ax.legend(fontsize=9, loc="lower right")
save(fig, p("fig20-06-lr-sweep.png"))

# 07 starts
names = [("scaled", "scaledx1@0.1", "adam 3e-3 warmup+cosine, scaled start"), ("default", "gpt2x1@0.1", "adam 3e-3 warmup+cosine"),
         ("fan-in", "fan_inx1@0.1", "adam 3e-3 warmup+cosine, fan-in start"), ("×10", "gpt2x10@0.1", "adam 3e-3 warmup+cosine, x10 start"),
         ("layer tables zero", "zerosx1@0.1", "adam 3e-3 warmup+cosine, layer tables zero")]
fig, ax = plt.subplots(figsize=(8.6, 4.4))
xx = np.arange(len(names))
a = [C19[k]["held_end"] for _, k, _ in names]; b = [TR[k]["held_end"] for _, _, k in names]
ax.bar(xx - 0.2, a, 0.38, color=VERM, label="plain GD, lr 0.1 (Chapter 19)"); ax.bar(xx + 0.2, b, 0.38, color=BLUE, label="Adam, lr 0.003, warmup + cosine")
for i in range(len(names)):
    ax.text(xx[i] - 0.2, a[i] + 0.02, f"{a[i]:.2f}", ha="center", fontsize=8.5); ax.text(xx[i] + 0.2, b[i] + 0.02, f"{b[i]:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(xx); ax.set_xticklabels([n for n, _, _ in names]); ax.set_ylim(4.8, 6.5); ax.set_ylabel("held-out loss after 400 steps")
ax.set_title(f"Adam narrows the gap between live starts from {max(a[:4]) - min(a[:4]):.2f} to {max(b[:4]) - min(b[:4]):.2f}"); ax.legend(fontsize=9, loc="upper left")
save(fig, p("fig20-07-starts.png"))

# 08 train vs held-out
fig, ax = plt.subplots(figsize=(8.6, 4.4))
for key, col, name in (("adam 3e-3 warmup+cosine", BLUE, "Adam"), ("muon 3e-3 warmup+cosine", PLUM, "Muon")):
    tr = np.array(TR[key]["train"]); sm = np.convolve(tr, np.ones(25) / 25, mode="valid")
    ax.plot(np.arange(24, 400), sm, color=col, lw=1.5, ls="--", label=f"{name}, batch loss (25-step average)")
    s_, h_ = held(key); ax.plot(s_, h_, color=col, marker="o", ms=3, label=f"{name}, held-out")
ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Muon learns the novel faster than it learns to generalise, lr 0.003"); ax.legend(fontsize=9)
save(fig, p("fig20-08-gap.png"))

# 09 memory
models = [("Tiny", 2_766_592), ("GPT-2 small", 124_439_808), ("SmolLM2-135M", 134_515_008), ("Qwen3-0.6B", 596_049_920), ("7B", 7_000_000_000)]
fig, ax = plt.subplots(figsize=(8.6, 4.4))
xx = np.arange(len(models)); w = np.array([n * 4 / 1e6 for _, n in models])
ax.bar(xx - 0.27, w, 0.26, color=NAVY, label="weights"); ax.bar(xx, w, 0.26, color=AMBER, label="nudges"); ax.bar(xx + 0.27, 2 * w, 0.26, color=PLUM, label="Adam's m and v")
for i in range(len(models)):
    ax.text(xx[i], 2 * w[i] * 1.35, f"total {4 * w[i]:,.0f} MB" if w[i] > 100 else f"total {4 * w[i]:.1f} MB", ha="center", fontsize=8.5, color=SLATE)
ax.set_yscale("log"); ax.set_xticks(xx); ax.set_xticklabels([m for m, _ in models]); ax.set_ylabel("megabytes, 32-bit (log)")
ax.set_title("Adam's state is half of training memory before activations"); ax.legend(fontsize=9, loc="upper left")
save(fig, p("fig20-09-memory.png"))

# 10 optim.py map
fig, ax = canvas(12, 5)
box(ax, 4, 18, 26, 14, "train_opt", NAVY, sub="nudges, clip, opt.step", fs=12)
box(ax, 40, 34, 24, 11, "SGD", VERM, sub="momentum buffer b", fs=11)
box(ax, 40, 19, 24, 11, "Adam / AdamW", BLUE, sub="m, v, bias correction, decay", fs=11)
box(ax, 40, 4, 24, 11, "Muon", PLUM, sub="Newton-Schulz; Adam for the rest", fs=10)
box(ax, 74, 19, 42, 11, "schedules", TEAL, sub="constant, warmup_cosine, wsd, linear_decay", fs=11)
for y in (39.5, 24.5, 9.5):
    arrow(ax, 30.5, 25, 39.5, y, color=GREY)
arrow(ax, 73.5, 24.5, 64.5, 24.5, color=TEAL)
label(ax, 60, 48, "tiny/optim.py: one line of Chapter 17's loop, replaced", color=NAVY, fs=13, weight="bold")
save(fig, p("fig20-10-optim-map.png"))
print("10 figures written to", OUT)
