"""Figures for Chapter 19, Initialization and Why Training Is Stable.  Every number is read from
04-code/ch19/outputs/*.json (run the chapter's scripts first, in the order of 04-code/ch19/RUN.txt).
Run from the project root:  python3 03-figures/src/ch19_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch19"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch19", "outputs")
js = lambda n: json.load(open(os.path.join(CH, n + ".json")))
p = lambda n: os.path.join(OUT, n)
IL, CHN, ZR, TL, RD, NF, LG = (js(n) for n in ("interlude", "chain", "zeros", "tiny_layers", "residual_depth", "norm_floor", "logits"))
minus = lambda s: s.replace("-", "−")

# 01 sums of coin flips
ns = [int(k) for k in IL["coins"]]; sd = [IL["coins"][k][0] for k in IL["coins"]]
fig, ax = plt.subplots(figsize=(8.4, 4.4))
xx = np.linspace(1, 2048, 400)
ax.plot(xx, np.sqrt(xx), color=GREY, lw=1.5, ls="--", label="√n")
ax.plot(ns, sd, "o", color=CORAL, ms=8, label="measured spread of the sum (100,000 sums)")
for n, s in zip(ns, sd):
    if n >= 16: ax.annotate(f"{s:.2f}", (n, s), textcoords="offset points", xytext=(6, -12), fontsize=9, color=SLATE)
ax.set_xscale("log", base=2); ax.set_xlabel("n, how many coin flips (+1 or −1) are added"); ax.set_ylabel("spread of the sum")
ax.set_title("Adding n random numbers multiplies the spread by √n"); ax.legend(fontsize=9)
save(fig, p("fig19-02-sums.png"))

# 02 and 03 the 48-layer chain, forward and backward
def chainplot(key, ttl, fname, ylabel):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    cols = {"0.02": PLUM, "1/sqrt(128)": BLUE, "sqrt(2)/sqrt(128)": TEAL, "0.2": VERM}
    nice = {"0.02": "0.02", "1/sqrt(128)": "1/√128", "sqrt(2)/sqrt(128)": "√2/√128", "0.2": "0.2"}
    for ax, act in zip(axs, ("linear", "relu", "gelu")):
        for name, col in cols.items():
            v = np.array(CHN[f"{act}|{name}"][key], dtype=float)
            v = np.where((v > 0) & np.isfinite(v), v, np.nan)
            ax.plot(range(len(v)), v, color=col, label=f"spread {nice[name]}")
        ax.set_yscale("log"); ax.set_title({"linear": "no activation", "relu": "ReLU", "gelu": "GELU"}[act], fontsize=12)
        ax.set_xlabel("layer"); ax.axhline(1, color=SLATE, lw=0.8)
    axs[0].set_ylabel(ylabel); axs[0].legend(fontsize=8.5)
    fig.suptitle(ttl, x=0.02, ha="left", fontsize=13, color=NAVY, weight="bold")
    save(fig, p(fname))
chainplot("forward", "A plain chain of 48 layers: the spread of the rows going up", "fig19-03-chain-forward.png", "spread of the rows (log scale)")
chainplot("backward", "The same chain: the spread of the nudge coming down (index = layer it enters)", "fig19-04-chain-backward.png", "spread of the nudge (log scale)")

# 04 symmetry: one neuron
fig, ax = canvas(11, 4.4)
for i in range(6):
    box(ax, 4 + i * 12.5 + (5 if i == 5 else 0), 26, 10, 9, "0.02", PLUM, sub="neuron %d" % (i + 1) if i < 5 else "neuron 512", fs=11)
label(ax, 70.5, 30.5, "…", fs=16)
box(ax, 4, 6, 36, 10, "same input, same output", NAVY, sub="every neuron computes one number", fs=11)
box(ax, 48, 6, 36, 10, "same nudge, same change", VERM, sub="they stay equal after every step", fs=11)
arrow(ax, 40.5, 11, 47.5, 11, color=GREY)
label(ax, 100, 30, f"Tiny, constant start:\n{ZR['constant']['distinct_W1']} distinct neuron of 512\nafter 20 steps", color=SLATE, fs=10)
label(ax, 100, 11, "a normalised row adds to 0,\nso a constant table\nadds exactly 0", color=SLATE, fs=10)
label(ax, 55, 41, "A constant start is one neuron copied 512 times", color=NAVY, fs=13, weight="bold")
save(fig, p("fig19-01-symmetry.png"))

# 05 Tiny's stream by layer under the starts
fig, ax = plt.subplots(figsize=(8.8, 4.6))
cols = [CORAL, BLUE, TEAL, AMBER, PLUM]
for (k, d), c in zip(TL.items(), cols):
    ax.plot(range(13), d["stream"], "o-", color=c, ms=4, label=d["desc"])
ax.set_yscale("log"); ax.set_xlabel("layer the rows enter (12 = leaving the last layer)"); ax.set_ylabel("spread of the stream (log scale)")
ax.set_title("Tiny's residual stream, layer by layer, under five starts"); ax.legend(fontsize=8.5)
save(fig, p("fig19-06-tiny-stream.png"))

# 06 residual depth
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3))
for L, c in zip((12, 24, 48, 96), (CORAL, BLUE, TEAL, PLUM)):
    for sch, ls in (("gpt2", "-"), ("scaled", "--")):
        s = RD[f"{L}|{sch}"]["stream"]
        axs[0].plot(np.arange(len(s)) / (len(s) - 1), s, ls, color=c, label=f"{L} layers, {'0.02' if sch == 'gpt2' else 'scaled'}")
axs[0].set_xlabel("fraction of the way up the stack"); axs[0].set_ylabel("spread of the stream"); axs[0].legend(fontsize=7.5, ncol=2)
axs[0].set_title("Unscaled branches pile up as √L", fontsize=12)
Ls = [12, 24, 48, 96]
g = [RD[f"{L}|gpt2"]["stream"][-1] / RD[f"{L}|gpt2"]["stream"][0] for L in Ls]
sc = [RD[f"{L}|scaled"]["stream"][-1] / RD[f"{L}|scaled"]["stream"][0] for L in Ls]
axs[1].plot(Ls, g, "o-", color=CORAL, label="every table 0.02"); axs[1].plot(Ls, sc, "s--", color=BLUE, label="W_o, W_2 divided by √(2L)")
for L, v in zip(Ls, g): axs[1].annotate(f"{v:.1f}×", (L, v), textcoords="offset points", xytext=(4, 6), fontsize=9)
for L, v in zip(Ls, sc): axs[1].annotate(f"{v:.2f}×", (L, v), textcoords="offset points", xytext=(4, 6), fontsize=9)
axs[1].set_xscale("log", base=2); axs[1].set_xticks(Ls, [str(L) for L in Ls]); axs[1].set_xlabel("layers"); axs[1].set_ylabel("growth, top over bottom")
axs[1].set_title("Scaling the writers by depth holds it at about 2×", fontsize=12); axs[1].legend(fontsize=9)
save(fig, p("fig19-05-residual-depth.png"))

# 07 nudges by layer
fig, ax = plt.subplots(figsize=(8.8, 4.6))
for (k, d), c in zip(TL.items(), cols):
    ax.plot(range(13), d["nudge"], "o-", color=c, ms=4, label=d["desc"])
ax.set_yscale("log"); ax.set_xlabel("layer the rows enter"); ax.set_ylabel("spread of the nudge on the stream (log)")
ax.set_title("The nudges on Tiny's stream: a larger stream receives smaller nudges"); ax.legend(fontsize=8.5)
save(fig, p("fig19-07-tiny-nudges.png"))

# 08 attention entropy against table size, plain and QK-norm
ks = [int(k) for k in LG]; sp = [0.02 * k for k in ks]
fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.plot(sp, [LG[str(k)]["plain"][1] for k in ks], "o-", color=VERM, label="plain: entropy of the shares")
ax.plot(sp, [LG[str(k)]["qknorm"][1] for k in ks], "s--", color=BLUE, label="QK-norm: entropy of the shares")
ax.axhline(3.909, color=GREY, lw=1, ls=":"); ax.text(0.03, 3.95, "uniform shares, 3.909", fontsize=9, color=GREY)
for k in ks:
    ax.annotate(f"scores {LG[str(k)]['plain'][0]:.3g}", (0.02 * k, LG[str(k)]["plain"][1]), textcoords="offset points", xytext=(5, 5), fontsize=8.5, color=VERM)
ax.set_xscale("log"); ax.set_xlabel("spread of W_q and W_k (log scale)"); ax.set_ylabel("average entropy of the shares")
ax.set_title("Layer 0, head 0: bigger query and key tables collapse attention"); ax.legend(fontsize=9, loc="lower left")
save(fig, p("fig19-10-attention-entropy.png"))

# 11 the norm floor
rs = [float(k) for k in NF]
fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.plot(rs, [NF[k][0] for k in NF], "o-", color=CORAL, label="RMSNorm, eps 1e−5")
ax.plot(rs, [NF[k][1] for k in NF], "s--", color=PLUM, label="RMSNorm, eps 1e−6")
ax.plot(rs, 1 / np.array(rs), color=GREY, lw=1, ls=":", label="1 / row spread")
ax.axhline(316.2, color=CORAL, lw=0.8, ls=":"); ax.axhline(1000, color=PLUM, lw=0.8, ls=":")
ax.axvline(0.02, color=TEAL, lw=1); ax.text(0.022, 2, "Tiny's token rows, 0.02", color=TEAL, fontsize=9)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.5, 3000); ax.invert_xaxis()
ax.set_xlabel("spread of the row entering the norm (log, smaller to the right)"); ax.set_ylabel("nudge leaving ÷ nudge arriving")
ax.set_title("A norm multiplies the nudge by 1/size, up to the floor 1/√eps"); ax.legend(fontsize=9, loc="upper left")
save(fig, p("fig19-11-norm-floor.png"))

# 12 init.py map
fig, ax = canvas(12, 5)
box(ax, 4, 32, 30, 11, "the dice", SLATE, sub="Tiny's own standard normal draws", fs=12)
schemes = [("gpt2", "0.02 everywhere"), ("scaled", "W_o, W_2 ÷ √(2L)"), ("fan_in", "1/√fan-in"), ("fan_in_scaled", "both"),
           ("zeros", "layer tables 0"), ("constant", "one number")]
for i, (n, s_) in enumerate(schemes):
    x = 42 + (i % 3) * 26; y = 34 if i < 3 else 20
    box(ax, x, y, 23, 10, n, [BLUE, TEAL, AMBER, PLUM, GREY, VERM][i], sub=s_, fs=11)
arrow(ax, 34.5, 37.5, 41.5, 38.5, color=GREY); arrow(ax, 34.5, 36, 41.5, 25, color=GREY)
box(ax, 42, 4, 75, 10, "× scale", CORAL, sub="every layer table times one number (10 = ten times too large)", fs=11)
box(ax, 4, 4, 30, 11, "spreads(m)", NAVY, sub="what each table now holds", fs=11)
label(ax, 60, 48, "tiny/init.py: change the size, keep the dice", color=NAVY, fs=13, weight="bold")
save(fig, p("fig19-12-init-map.png"))

# 09 and 10 need the training runs
if os.path.exists(os.path.join(CH, "train_inits.json")):
    TR = js("train_inits")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, lr in zip(axs, (0.1, 0.6)):
        for (k, d), c in zip([(k, d) for k, d in TR.items() if d["lr"] == lr], cols):
            cv = np.array(d["curve"])
            if len(cv): ax.plot(cv[:, 0], cv[:, 2], "o-", ms=3, color=c, label=d["desc"])
        ax.set_title(f"learning rate {lr}", fontsize=12); ax.set_xlabel("step")
    axs[0].set_ylabel("held-out loss (16 windows)"); axs[0].legend(fontsize=8.5)
    fig.suptitle("Training Tiny from five starts, plain gradient descent, 400 steps", x=0.02, ha="left", fontsize=13, color=NAVY, weight="bold")
    save(fig, p("fig19-08-training.png"))
if os.path.exists(os.path.join(CH, "clipping.json")):
    CL = js("clipping")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for key, c, nm in (("None", VERM, "no clipping"), ("1.0", BLUE, "clip at 1.0"), ("5.0", TEAL, "clip at 5.0")):
        d = CL[key]
        axs[0].plot(range(len(d["train"])), d["train"], color=c, lw=1.4, label=nm)
        axs[1].plot(range(len(d["norms"])), d["norms"], color=c, lw=1.2, label=nm)
    axs[0].set_yscale("log"); axs[0].set_xlabel("step"); axs[0].set_ylabel("batch loss (log scale)"); axs[0].set_title("Learning rate 1.0: the loss", fontsize=12); axs[0].legend(fontsize=9)
    axs[1].set_yscale("log"); axs[1].set_xlabel("step"); axs[1].set_ylabel("global norm of the nudges, before clipping")
    axs[1].set_title("The nudges' global norm", fontsize=12)
    save(fig, p("fig19-09-clipping.png"))
print("figures written to", OUT)
