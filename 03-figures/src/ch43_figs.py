"""Figures for Chapter 43, The Same Machine Looks at Pictures.  Numbers are read from 04-code/ch43/outputs/ (run 04-code/ch43/RUN.txt first).
Run from the project root: python3 03-figures/src/ch43_figs.py"""
import os, sys, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code", "ch43"))
from style import *
import matplotlib.pyplot as plt, numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch43"); os.makedirs(OUT, exist_ok=True)
O = os.path.join(ROOT, "04-code", "ch43", "outputs")
J = lambda n: json.load(open(os.path.join(O, n + ".json")))
p = lambda n: os.path.join(OUT, n)
import vit
_, _, Xte, yte = vit.data(); img = Xte[0].numpy(); P = vit.patches(Xte[:1])[0].numpy()

# 01 one digit cut into a 4 x 4 grid of patches, read like words
fig = plt.figure(figsize=(11.0, 4.0)); ax = fig.add_axes([0.02, 0.04, 0.30, 0.78])
ax.imshow(img, cmap="Greys", vmin=0, vmax=1)
for k in range(1, 4): ax.axhline(2 * k - 0.5, color=BLUE, lw=2); ax.axvline(2 * k - 0.5, color=BLUE, lw=2)
for i in range(16): ax.text((i % 4) * 2 + 0.5, (i // 4) * 2 + 0.5, str(i), color=VERM, fontsize=9, ha="center", va="center", weight="bold")
ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"a test image (label {int(yte[0])}), 8 x 8 pixels, cut into 16 patches", fontsize=9.5)
ax2 = fig.add_axes([0.36, 0.30, 0.62, 0.45])
strip = np.zeros((2, 16 * 3 - 1)) + np.nan
for i in range(16): strip[:, i * 3:i * 3 + 2] = P[i].reshape(2, 2)
ax2.imshow(np.ma.masked_invalid(strip), cmap="Greys", vmin=0, vmax=1)
for i in range(16): ax2.text(i * 3 + 0.5, 2.2, str(i), ha="center", va="top", fontsize=8.5, color=VERM)
ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_frame_on(False)
ax2.set_title("the same image as a sequence of 16 patches (2 x 2 pixels each), left to right, top to bottom;\n"
              "each patch's 4 numbers go through one learned 4 x 64 table, as a token id goes through the embedding table", fontsize=9)
fig.suptitle("Patches as tokens", fontsize=11, x=0.01, ha="left")
save(fig, p("fig43-01-patches.png"))

# 02 accuracy by position scheme, in order and with patches shuffled at test time
runs = [("learned", ["vit", "vit_s2", "vit_s3"]), ("row + column", ["vit_rowcol", "vit_rowcol_s2"]), ("2-D sines", ["vit_sincos", "vit_sincos_s2"]),
        ("none", ["vit_nopos", "vit_nopos_s2"]), ("learned,\ntrained shuffled", ["vit_shuf"]), ("learned,\nmean pooling", ["vit_mean", "vit_mean_s2"])]
fig, ax = plt.subplots(figsize=(10.5, 4.2)); x = np.arange(len(runs))
a = [np.mean([J(n)["test"] for n in ns]) for _, ns in runs]; b = [np.mean([J(n)["test_shuffled"] for n in ns]) for _, ns in runs]
ax.bar(x - 0.2, a, 0.4, color=BLUE, label="patches in order"); ax.bar(x + 0.2, b, 0.4, color=VERM, label="patches shuffled at test time")
for i in range(len(runs)): ax.text(x[i] - 0.2, a[i] + 0.01, f"{a[i]:.3f}", ha="center", fontsize=8); ax.text(x[i] + 0.2, b[i] + 0.01, f"{b[i]:.3f}", ha="center", fontsize=8)
ax.axhline(0.1, color=GREY, ls=":", lw=1); ax.text(len(runs) - 0.5, 0.11, "chance 0.1", fontsize=8, color=GREY, ha="right")
ax.set_xticks(x); ax.set_xticklabels([r for r, _ in runs], fontsize=8.5); ax.set_ylabel("test accuracy, 360 images (mean of seeds run)"); ax.set_ylim(0, 1.08)
ax.legend(fontsize=8, loc="upper center", ncol=2); fig.suptitle("Position is what makes order matter", fontsize=11, x=0.01, ha="left")
save(fig, p("fig43-02-positions.png"))

# 03 accuracy against training images, Tiny-ViT against three baselines
fr = [0.1, 0.25, 0.5, 1.0]; n_img = [144, 359, 718, 1437]
v = [np.mean([J(f"vit_f{f}_s{s}")["test"] for s in (1865, 2)]) for f in fr[:3]] + [np.mean([J(n)["test"] for n in ("vit", "vit_s2", "vit_s3")])]
B = {f: J(f"baseline_s1865_f{f}") for f in fr[:3]}; B[1.0] = J("baseline_s1865")
fig, ax = plt.subplots(figsize=(8.5, 4.3))
ax.plot(n_img, v, "o-", color=BLUE, lw=2, label="Tiny-ViT (202,186 numbers; seeds 1865 and 2, three seeds at 1,437)")
for k, c, lab in (("cnn", TEAL, "small CNN (9,930)"), ("mlp", AMBER, "MLP (19,210)"), ("softmax", GREY, "one table (650)")):
    ax.plot(n_img, [B[f][k]["test"] for f in fr], "s--", color=c, lw=1.5, label=lab + ", seed 1865")
ax.set_xscale("log"); ax.xaxis.set_minor_locator(plt.NullLocator()); ax.set_xticks(n_img); ax.set_xticklabels([f"{n:,}" for n in n_img]); ax.set_xlabel("training images (same number of steps for every size)")
ax.set_ylabel("test accuracy, 360 images"); ax.legend(fontsize=7.5, loc="lower right"); ax.set_ylim(0.6, 1.0)
fig.suptitle("With little data, the network built for images wins", fontsize=11, x=0.01, ha="left")
save(fig, p("fig43-03-data.png"))

# 04 image tokens charged by a hosted model against image side
fig, ax = plt.subplots(figsize=(8.5, 4.3))
for f, c, lab in (("image_tokens_claude-haiku-4-5-20251001", BLUE, "Claude Haiku 4.5"), ("image_tokens_claude-haiku-5-5", VERM, "Claude Haiku 5.5")):
    R = J(f); s = [z["side"] for z in R["sizes"]]; t = [z["tokens"] for z in R["sizes"]]
    ax.plot(s, t, "o-", color=c, lw=1.8, label=lab + " (measured)")
ss = np.array([8, 32, 64, 128, 256, 512, 768, 1024, 1568, 1932]); ax.plot(ss, np.ceil(ss / 28) ** 2, ":", color=GREY, lw=1.4, label="ceil(side / 28) squared")
ax.set_xscale("log"); ax.set_yscale("log"); ax.xaxis.set_minor_locator(plt.NullLocator()); ax.set_xticks([8, 32, 128, 512, 2048]); ax.set_xticklabels(["8", "32", "128", "512", "2,048"]); ax.set_xlabel("side of a square image (pixels)"); ax.set_ylabel("input tokens for the image")
ax.legend(fontsize=8, loc="upper left"); fig.suptitle("An image's token bill is a patch count, up to a resize limit", fontsize=11, x=0.01, ha="left")
save(fig, p("fig43-04-tokens.png"))
print("ok")
