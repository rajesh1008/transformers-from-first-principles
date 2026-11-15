"""Figures for Chapter 47, The Machine-Learning Engineer's Interview, From Both Sides.
fig47-01 reads 04-code/ch47/outputs/rubric47.json; fig47-02 is a drawn diagram of the loop and the chapters behind each round.
Run from the project root: python3 03-figures/src/ch47_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt, numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch47"); os.makedirs(OUT, exist_ok=True); p = lambda n: os.path.join(OUT, n)
R = json.load(open(os.path.join(ROOT, "04-code", "ch47", "outputs", "rubric47.json")))["scores"]
names = list(R); x = np.arange(len(names)); w = 0.2
series = [("target level (as written)", [R[n]["target"] for n in names], GREY),
          ("keyword rubric", [R[n]["keyword"] for n in names], VERM),
          ("anchored rubric, grader mean", [np.mean(R[n]["anchored"]) for n in names], TEAL),
          ("vocabulary rubric, grader mean", [np.mean(R[n]["vocab"]) for n in names], AMBER)]
fig, ax = plt.subplots(figsize=(9.6, 4.8))
for i, (lab, v, col) in enumerate(series):
    ax.bar(x + (i - 1.5) * w, v, w, color=col, label=lab)
ax.set_xticks(x); ax.set_xticklabels([f"answer {n}" for n in names]); ax.set_ylim(0, 5.6); ax.set_yticks([0, 1, 2, 3, 4]); ax.set_ylabel("score (0 to 4)")
ax.annotate("jargon, wrong mechanism", xy=(1 - 0.5 * w, 4.0), xytext=(1.4, 4.35), fontsize=9, color=VERM, arrowprops=dict(arrowstyle="-", color=VERM))
ax.legend(fontsize=8.5, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.0))
fig.suptitle("Six synthetic answers to the whiteboard exercise, scored three ways", fontsize=11, x=0.01, ha="left"); save(fig, p("fig47-01-rubrics.png"))
# 02 the loop
fig, ax = canvas(10, 5.2)
rounds = [("Phone screen", "explain one mechanism\nin a minute", "Ch1-16"), ("Whiteboard", "derive attention, count\nparameters, cache bytes", "Ch8, 16, 30"),
          ("Debugging", "a loss curve or a\nflawed training change", "Ch17-26"), ("Systems design", "serve a model at a\nlatency and a cost", "Ch30-33, 45"),
          ("Paper discussion", "read a result aloud;\nwhat would you test", "every research box"), ("Past run", "present your own run\nand its reproducibility", "Ch26, Lab of choice")]
for i, (t, s, c) in enumerate(rounds):
    xx = 2 + (i % 3) * 32; yy = 52 if i < 3 else 10
    box(ax, xx, yy, 28, 30, t, color=SERIES[i % len(SERIES)], sub=s, fs=11)
    label(ax, xx + 14, yy - 3.5, c, fs=9)
    if i % 3 != 2: arrow(ax, xx + 28, yy + 15, xx + 32, yy + 15)
arrow(ax, 94, 67, 94, 40, rad=0.0)
ax.set_xlim(0, 100); ax.set_ylim(0, 90)
title(ax, "A machine-learning engineering loop, and the chapters behind each round", fs=12)
save(fig, p("fig47-02-loop.png"))
print("ok")
