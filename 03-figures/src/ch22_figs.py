"""Figures for Chapter 22, Data Pipelines and the Curriculum.  Every measured number is read from
04-code/ch22/outputs/*.json; run the chapter's scripts first, in the order of 04-code/ch22/RUN.txt.
fig22-02 draws the banding formula (no measurement).  Run from the project root: python3 03-figures/src/ch22_figs.py"""
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code"))
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch22"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch22", "outputs")
js = lambda n: json.load(open(os.path.join(CH, n + ".json")))
p = lambda n: os.path.join(OUT, n)
TD = {os.path.basename(f)[11:-5]: json.load(open(f)) for f in glob.glob(os.path.join(CH, "train_data_*.json"))}
PB, ST = js("pipeline_basics"), js("stream")

# 01 padded against packed seats
fig, ax = plt.subplots(figsize=(8.6, 4.2))
labs = ["chapters\npadded", "chapters\npacked", "paragraphs\npadded", "paragraphs\npacked"]
vals = [PB["seats_chapters"]["padded_waste"], 0.0016, PB["seats_paragraphs"]["padded_waste"], 0.0004]
ax.bar(labs, [100 * v for v in vals], color=[VERM, TEAL, VERM, TEAL])
for i, v in enumerate(vals):
    ax.text(i, 100 * v + 1.5, f"{100 * v:.2f}%" if v < 0.01 else f"{100 * v:.1f}%", ha="center", fontsize=10)
ax.set_ylabel("seats wasted on padding (%)"); ax.set_ylim(0, 75)
ax.set_title("Short documents in windows of 129: padding wastes most seats, packing almost none")
save(fig, p("fig22-01-seats.png"))

# 02 the banding S-curve
fig, ax = plt.subplots(figsize=(8.6, 4.2))
s = np.linspace(0, 1, 201)
for (b, r), col in zip([(32, 4), (16, 8), (64, 2), (20, 5)], SERIES):
    ax.plot(s, 1 - (1 - s ** r) ** b, color=col, lw=2.2, label=f"{b} bands of {r}")
ax.axvline(0.5, color=GREY, lw=1, ls="--")
ax.set_xlabel("Jaccard similarity of a pair"); ax.set_ylabel("chance the pair becomes a candidate")
ax.legend(frameon=False, loc="lower right"); ax.set_title("Banding 128 MinHash numbers: the bands and rows set the threshold")
save(fig, p("fig22-02-banding.png"))

# 03 repetition at a fixed budget
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.2))
for P, col in zip([1, 2, 4, 8, 32], SERIES):
    r = TD.get(f"rep{P}")
    if not r: continue
    c = np.array([(x[0], x[2], x[4]) for x in r["curve"]])
    a1.plot(c[:, 0], c[:, 1], color=col, lw=2, label=f"{P} pass{'es' if P > 1 else ''}")
    a2.plot(c[:, 0], c[:, 2], color=col, lw=2)
a1.set_title("held-out (Looking-Glass)"); a2.set_title("windows the run read")
for a in (a1, a2):
    a.set_xlabel("step"); a.set_ylim(3.0, 8.0)
a1.set_ylabel("loss"); a1.legend(frameon=False)
fig.suptitle("The same 384 steps on 1 to 32 passes of a slice: only 32 passes clearly hurt", y=1.03)
save(fig, p("fig22-03-repetition.png"))

# 04 mixtures
fig, ax = plt.subplots(figsize=(8.6, 4.2))
ws = [0, 10, 25, 50, 100]
for key, col, mk, lab in (("glass", BLUE, "o", "held-out Looking-Glass"), ("carroll", AMBER, "s", "held-out Carroll pool")):
    runs = [[TD[f"mix{w}"][key]] + ([TD[f"mix{w}_s7"][key]] if f"mix{w}_s7" in TD else []) for w in ws]
    ax.plot(ws, [np.mean(r) for r in runs], mk + "-", color=col, lw=2.2, label=lab + " (average of two seeds)")
    for w, r in zip(ws, runs):
        ax.plot([w] * len(r), r, mk, mfc="white", color=col, ms=6)
ax.set_xlabel("share of windows drawn from Alice (%)"); ax.set_ylabel("held-out loss after 384 steps")
ax.legend(frameon=False); ax.set_title("Two sources mixed, two seeds each: only Alice alone stands out, on the Carroll set")
save(fig, p("fig22-04-mixture.png"))

# 05 shuffle buffer
fig, ax = plt.subplots(figsize=(8.6, 4.2))
bs = [1, 8, 64, 256, 1024]
ax.semilogx(bs, [ST[f"buffer_{b}"]["same_book"] for b in bs], "o-", color=PLUM, lw=2.2, base=2)
FS = js("full_shuffle")["share"]
ax.axhline(FS, color=GREY, ls="--", lw=1)
ax.text(1.1, FS - 0.04, f"a full shuffle of the pool: {FS:.2f} (simulated)", fontsize=9, color=GREY)
ax.set_xlabel("shuffle buffer (windows)"); ax.set_ylabel("largest share of one book in a batch")
ax.set_ylim(0.3, 1.02); ax.set_title("A shuffle buffer mixes only what it holds")
save(fig, p("fig22-05-buffer.png"))

# 06 curricula
fig, ax = plt.subplots(figsize=(8.6, 4.2))
for k, col, lab in [("rep1", BLUE, "random order"), ("cur_easy", TEAL, "easy first"), ("cur_hard", VERM, "hard first"),
                    ("cur_inorder", GREY, "book order (no shuffle)")]:
    if k not in TD: continue
    c = np.array([(x[0], x[3]) for x in TD[k]["curve"]])
    ax.plot(c[:, 0], c[:, 1], color=col, lw=2, label=lab)
ax.set_xlabel("step"); ax.set_ylabel("held-out loss (Carroll pool)"); ax.set_ylim(5.0, 8.0)
ax.legend(frameon=False); ax.set_title("The same 3,072 windows in four orders, warmup and cosine decay")
save(fig, p("fig22-06-curriculum.png"))
print("ok")
