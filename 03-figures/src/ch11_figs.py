"""Figures for Chapter 11, Attention Efficiency.  Every number is read from 04-code/ch11/outputs or
recomputed here from the chapter's own code.  Run from the project root: python3 03-figures/src/ch11_figs.py"""
import os, sys, re, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib, matplotlib.pyplot as plt
setup()
OUT = os.path.join(ROOT, "03-figures", "ch11"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch11", "outputs")

# 01 traffic: plain vs tiled bytes
txt = open(os.path.join(CH, "traffic.txt")).read()
rows = re.findall(r"^\s*([\d,]+) \|\s+([\d,]+) \|\s+([\d,]+) \|\s+([\d,]+) \|", txt, re.M)
ns = [int(a.replace(",", "")) for a, *_ in rows]
plain = [int(c.replace(",", "")) / 2**20 for _, _, c, _ in rows]; tiled = [int(dd.replace(",", "")) / 2**20 for *_, dd in rows]
fig, ax = plt.subplots(figsize=(8.4, 4.4))
x = np.arange(len(ns)); w = 0.38
ax.bar(x - w / 2, plain, w, color=TINT[VERM], edgecolor=VERM, label="plain: scores and shares go to memory")
ax.bar(x + w / 2, tiled, w, color=TINT[TEAL], edgecolor=TEAL, label="tiled: Q, K, V in, output out")
ax.set_yscale("log"); ax.set_xticks(x, [f"{n:,}" for n in ns]); ax.set_xlabel("tokens"); ax.set_ylabel("MiB moved, one head (log scale)")
for i in range(len(ns)):
    ax.text(x[i] - w / 2, plain[i] * 1.15, f"{plain[i]:,.0f}", ha="center", fontsize=8.5, color=VERM)
    ax.text(x[i] + w / 2, tiled[i] * 1.15, f"{tiled[i]:,.0f}", ha="center", fontsize=8.5, color=TEAL)
ax.set_title("Bytes moved by one head of width 64, 16-bit"); ax.legend(loc="upper left")
save(fig, os.path.join(OUT, "fig11-01-traffic.png"))

# 02 online softmax trace on mat's row
s = np.array([1.414, -0.707, 0.707, 2.121, 1.414, -2.121]); words = ["the", "cat", "sat", "on", "the", "mat"]
txt = open(os.path.join(CH, "online_softmax.txt")).read()
mx = [float(a) for a in re.findall(r"running max\s+([-\d.]+)", txt)]; sm = [float(a) for a in re.findall(r"running sum ([\d.]+)", txt)]
fig, ax = plt.subplots(figsize=(8.4, 4.2))
ax.bar(range(6), s, color=TINT[PLUM], edgecolor=PLUM, label="score")
ax.step(range(6), mx, where="mid", color=NAVY, label="running max")
ax2 = ax.twinx(); ax2.plot(range(6), sm, marker="o", color=TEAL, label="running sum"); ax2.set_ylabel("running sum"); ax2.grid(False); ax2.set_ylim(0, 2.6)
ax.annotate("max grows: old sum x 0.493", (3, 2.121), (3.4, 2.6), fontsize=9.5, color=CORAL, arrowprops=dict(arrowstyle="-|>", color=CORAL))
ax.set_xticks(range(6), words); ax.set_ylim(-2.5, 3.0); ax.set_ylabel("score")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="lower left")
ax.set_title("Online softmax along mat's row: one pass, one rescale")
save(fig, os.path.join(OUT, "fig11-02-online.png"))

# 03 tiles computed and skipped, causal and window 256, n = 1,024 in 64 x 64 blocks
T = 16
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for ax, (t, win) in zip(axs, [("causal: 136 of 256 blocks", None), ("causal + window 256: 70 of 256", 256)]):
    G = np.zeros((T, T)); r = np.arange(1024)
    for i in range(T):
        for j in range(T):
            rows, cols = r[i * 64:(i + 1) * 64, None], r[None, j * 64:(j + 1) * 64]
            ok = cols <= rows
            if win: ok = ok & (cols > rows - win)
            G[i, j] = 0 if not ok.any() else (2 if ok.all() else 1)
    cmap = matplotlib.colors.ListedColormap(["#f3f4f6", TINT[AMBER], TINT[BLUE]])
    ax.imshow(G, cmap=cmap, vmin=0, vmax=2)
    for i in range(T + 1):
        ax.axhline(i - .5, color="white", lw=1); ax.axvline(i - .5, color="white", lw=1)
    ax.set_xticks(range(0, 16, 3)); ax.set_yticks(range(0, 16, 3)); ax.set_title(t, fontsize=11); ax.set_xlabel("key block"); ax.set_ylabel("query block"); ax.grid(False)
fig.text(0.5, -0.02, "grey: skipped   amber: partly masked   blue: computed in full", ha="center", fontsize=10, color=SLATE)
save(fig, os.path.join(OUT, "fig11-03-tiles.png"))

# 04 where the CPU time goes, plain attention n = 2,048
r = json.load(open(os.path.join(CH, "roofline.json")))
k = list(r["attn_ms"].keys()); v = [r["attn_ms"][a] for a in k]
fig, ax = plt.subplots(figsize=(8.4, 3.8))
cols = [BLUE, AMBER, PLUM, PLUM, BLUE]
ax.barh(range(len(k)), v, color=[TINT[c] for c in cols], edgecolor=cols)
for i, a in enumerate(v):
    ax.text(a + 0.05, i, f"{a:.2f} ms", va="center", fontsize=9.5, color=NAVY)
ax.set_yticks(range(len(k)), k); ax.invert_yaxis(); ax.set_xlabel("milliseconds (fastest of 5)")
ax.set_title("One head, 2,048 tokens on the book's CPU: the multiplies are the cheap part")
save(fig, os.path.join(OUT, "fig11-04-cpu-steps.png"))

# 05 tiled vs plain time
txt = open(os.path.join(CH, "tiled.txt")).read()
tm = re.findall(r"n=\s*(\d+): plain\s+([\d.]+) \| tiled 64 x 64\s+([\d.]+) \| tiled 256 x 256\s+([\d.]+)", txt)
fig, ax = plt.subplots(figsize=(8.4, 4.2))
nn = [int(a) for a, *_ in tm]
for idx, (lab, col) in enumerate([("plain (whole table)", VERM), ("tiled 64 x 64", TEAL), ("tiled 256 x 256", BLUE)]):
    ax.plot(nn, [float(t[idx + 1]) for t in tm], marker="o", color=col, label=lab)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(nn, [f"{a:,}" for a in nn])
ax.set_xlabel("tokens"); ax.set_ylabel("ms, one causal head (log)"); ax.legend(); ax.set_title("NumPy on a CPU: tiling wins once the table outgrows the caches")
save(fig, os.path.join(OUT, "fig11-05-tiled-time.png"))
print("ch11 figures written to", OUT)

# 06 the three shelves: reading speed against array size (shelves.py)
r = json.load(open(os.path.join(CH, "shelves.json")))
kib = [16, 32, 256, 1024, 4096, 32768, 262144, 1048576]
gbs = [r[str(k)] for k in kib]
fig, ax = plt.subplots(figsize=(8.4, 5.6))
ax.plot(kib, gbs, marker="o", color=NAVY, lw=2.2)
for x0, x1, name, col in [(10, 2048, "first and second shelves:\neach core's caches (2 MiB per core)", TEAL),
                          (2048, 327680, "third shelf: shared cache\n(320 MiB)", AMBER),
                          (327680, 2.2e6, "main memory", VERM)]:
    ax.axvspan(x0, x1, color=TINT.get(col, LIGHT), alpha=1.0, zorder=0)
    ax.text((x0 * x1) ** 0.5, 67, name, ha="center", va="top", fontsize=9, color=col)
for k, g in zip(kib, gbs):
    ax.text(k, g - 5.5, f"{g:.0f}", ha="center", fontsize=8.5, color=SLATE)
ax.set_xscale("log"); ax.set_xlim(10, 2.2e6); ax.set_ylim(0, 72)
ax.set_xticks(kib, ["16K", "32K", "256K", "1M", "4M", "32M", "256M", "1G"])
ax.set_xlabel("size of the array read, bytes (K = KiB, M = MiB, G = GiB; log scale)"); ax.set_ylabel("GB per second, fastest of 5")
ax.set_title("Reading speed on the book's CPU falls at each shelf")
save(fig, os.path.join(OUT, "fig11-06-shelves.png"))

# 07 Lab 11: the 6 x 6 share table rebuilt from two 3-wide tiles
txt = open(os.path.join(CH, "lab11.txt")).read()
blk = txt.split("recovered shares")[1].split("equal to")[0].splitlines()[2:8]
P = np.array([[float(v) for v in line.split()[1:]] for line in blk])
words = ["the", "cat", "sat", "on", "the", "mat"]
fig, ax = plt.subplots(figsize=(6.6, 6.2))
ax.imshow(np.where(P > 0, P, np.nan), cmap="Blues", vmin=0, vmax=1.2)
for i in range(6):
    for j in range(6):
        if j <= i:
            ax.text(j, i, f"{P[i, j]:.3f}", ha="center", va="center", fontsize=9.5, color="white" if P[i, j] > 0.6 else NAVY)
        else:
            ax.text(j, i, "skipped" if (i < 3 and j >= 3) else "masked", ha="center", va="center", fontsize=7.5, color=GREY)
for c in (2.5,):
    ax.axhline(c, color=VERM, lw=2.5); ax.axvline(c, color=VERM, lw=2.5)
ax.set_xticks(range(6), words); ax.set_yticks(range(6), words)
ax.set_xlabel("key tile 0 (seats 0 to 2) | key tile 1 (seats 3 to 5)"); ax.set_ylabel("query tile 0 | query tile 1")
ax.set_title("Two 3-wide tiles rebuild Chapter 8's table")
ax.grid(False)
save(fig, os.path.join(OUT, "fig11-07-lab-tiles.png"))

# 08 Lab 11 breakages: mat's output, correct and broken
vals = {"correct": (0.665, -0.297), "Breakage 1: nothing rescaled": (0.833, -0.119), "Breakage 2: sum only": (1.129, -0.162)}
fig, ax = plt.subplots(figsize=(7.6, 5.6))
offs = {"correct": (-0.05, -0.05, "top"), "Breakage 1: nothing rescaled": (-0.35, 0.16, "bottom"), "Breakage 2: sum only": (0.0, -0.06, "top")}
for (name, (x, y)), col, ls in zip(vals.items(), [TEAL, VERM, CORAL], ["-", "--", ":"]):
    ax.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2, ls=ls))
    dx, dy, va = offs[name]
    ax.text(x + dx, y + dy, f"{name}\n({x}, {str(y).replace('-', '\u2212')})", fontsize=9.5, color=col, va=va)
ax.axhline(0, color=GREY, lw=0.75); ax.axvline(0, color=GREY, lw=0.75)
ax.set_xlim(-0.1, 1.6); ax.set_ylim(-0.5, 0.25)
ax.set_xlabel("first number of mat's output"); ax.set_ylabel("second number")
ax.set_title("mat's output, tiles of 3: right and wrong")
save(fig, os.path.join(OUT, "fig11-08-breakages.png"))
