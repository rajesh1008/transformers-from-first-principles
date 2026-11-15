"""Figures for Chapter 37, Why It Makes Things Up.  Numbers are read from 04-code/ch37/outputs/ (run 04-code/ch37/RUN.txt
first).  Run from the project root: python3 03-figures/src/ch37_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch37"); os.makedirs(OUT, exist_ok=True)
O = os.path.join(ROOT, "04-code", "ch37", "outputs")
J = lambda n: json.load(open(os.path.join(O, n + ".json")))
p = lambda n: os.path.join(OUT, n)

# 01 reliability diagram: confidence against accuracy by bin, three models
d = J("calibration")
fig, ax = plt.subplots(figsize=(7.2, 6.0))
ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
ax.annotate("perfectly calibrated", (0.62, 0.66), rotation=40, color=GREY, fontsize=9)
for (name, col, mk) in (("Tiny", TEAL, "o"), ("GPT-2 small", VERM, "s"), ("SmolLM2-135M", BLUE, "D")):
    if name not in d: continue
    B = [b for b in d[name]["bins"] if b[1]]
    ax.plot([b[2] for b in B], [b[3] for b in B], marker=mk, color=col, label=f"{name} (ECE {d[name]['ece']:.3f})")
ax.set_xlabel("mean top-1 share in the bin (confidence)"); ax.set_ylabel("share of top choices that were right (accuracy)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False, loc="upper left")
ax.set_title("Through the Looking-Glass, about 20,000 predictions per model", fontsize=11)
save(fig, p("fig37-01-calibration.png"))

# 02 capitals: confidence in the written sentence, right against not
d = J("seen")
fig, ax = plt.subplots(figsize=(8.4, 4.8))
xs = {"GPT-2 small": 0, "SmolLM2-135M": 1}
for lab, x0 in xs.items():
    for grp, dx in (("well-known", -0.15), ("less-known", 0.15)):
        jit = np.random.default_rng(1865).uniform(-0.05, 0.05, size=len(d[lab][grp]["rows"]))
        for (cn, txt, v, ok), j in zip(d[lab][grp]["rows"], jit):
            ax.scatter(x0 + dx + j, v, color=GREEN if ok else VERM, s=30, alpha=0.85,
                       marker="o" if grp == "well-known" else "s", edgecolor="white", linewidth=0.5)
    for cn, txt, v in d[lab]["invented"]:
        ax.scatter(x0 + 0.42, v, color=PLUM, s=30, marker="^")
ax.set_xticks([-0.15, 0.15, 0.42, 0.85, 1.15, 1.42]); ax.set_xticklabels(["GPT-2\nwell-known", "GPT-2\nless-known", "GPT-2\ninvented", "SmolLM2\nwell-known", "SmolLM2\nless-known", "SmolLM2\ninvented"], fontsize=9)
ax.set_ylabel("mean share of the tokens it wrote")
ax.scatter([], [], color=GREEN, label="names the real capital"); ax.scatter([], [], color=VERM, label="does not"); ax.scatter([], [], color=PLUM, marker="^", label="invented country")
ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.set_title('"The capital of X is": confidence does not separate right from wrong', fontsize=11)
save(fig, p("fig37-02-capitals.png"))

# 03 Tiny: seen often against seen once (tiny_world.py, part A)
d = J("tiny_world")["A"]
fig, ax = plt.subplots(figsize=(7.6, 4.8))
bk = ["1", "2-4", "5-19", "20+"]; x = np.arange(4)
ax.bar(x - 0.2, [d[b]["acc"] for b in bk], 0.38, color=TEAL, label="top choice right (accuracy)")
ax.bar(x + 0.2, [d[b]["conf"] for b in bk], 0.38, color=AMBER, label="top-1 share (confidence)")
for i, b in enumerate(bk):
    ax.text(i - 0.2, d[b]["acc"] + 0.02, f"{d[b]['acc']:.2f}", ha="center", fontsize=9)
    ax.text(i + 0.2, d[b]["conf"] + 0.02, f"{d[b]['conf']:.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f"seen {b} time{'s' if b != '1' else ''}\n(n = {d[b]['n']})" for b in bk])
ax.set_ylim(0, 1.1); ax.set_ylabel("share"); ax.legend(frameon=False, loc="upper left")
ax.set_title("Tiny on its own training text: how often a 5-token context occurred", fontsize=11)
save(fig, p("fig37-03-seen-once.png"))

# 05 agreement between ten samples against how many were right (agree2.py), SmolLM2-135M
d = J("agree2")
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
for ax, key in zip(axs, ("SmolLM2-135M T=1.0", "SmolLM2-135M T=0.7")):
    rows = [r for r in d[key] if r["group"] != "invented"]
    jit = np.random.default_rng(1865).uniform(-0.012, 0.012, size=len(rows))
    for r, j in zip(rows, jit):
        ax.scatter(r["agree"] + j, r["right"], color=GREEN if r["majority_right"] else VERM, s=34, alpha=0.85,
                   marker="o" if r["group"] == "well" else "s", edgecolor="white", linewidth=0.5)
    for r in rows:
        if r["country"] in ("Moldova", "Thailand", "Spain", "Tajikistan"):
            ax.annotate(r["country"], (r["agree"], r["right"]), xytext=(5, 3), textcoords="offset points", fontsize=8, color=GREY)
    ax.set_title(key.replace("T=", "temperature "), fontsize=10); ax.set_xlabel("agreement (share of the ten for the commonest answer)", fontsize=9)
    ax.set_xlim(-0.05, 1.0)
axs[0].set_ylabel("samples naming the real capital (of 10)")
axs[1].scatter([], [], color=GREEN, label="commonest answer is the capital"); axs[1].scatter([], [], color=VERM, label="it is not")
axs[1].legend(frameon=False, fontsize=8, loc="upper left")
save(fig, p("fig37-05-agreement.png"))

# 06 where a greedy answer's confidence sits: whole sentence against the name token (name_share.py), SmolLM2-135M
d = J("name_share")["SmolLM2-135M"]
fig, ax = plt.subplots(figsize=(7.2, 5.6))
ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
for r in d:
    if r["name_share"] is None: continue
    col = PLUM if r["group"] == "invented" else (GREEN if r["right"] else VERM)
    ax.scatter(r["mean_share"], r["name_share"], color=col, s=34, marker="^" if r["group"] == "invented" else "o", edgecolor="white", linewidth=0.5)
    if r["country"] in ("Moldova", "Tajikistan", "Laos", "Mongolia", "Zubrovia"):
        ax.annotate(r["country"], (r["mean_share"], r["name_share"]), xytext=(5, -10 if r["country"] == "Tajikistan" else 3), textcoords="offset points", fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("mean share of all ten tokens written (the sentence)"); ax.set_ylabel("share on the first token of the name")
ax.scatter([], [], color=GREEN, label="names the real capital"); ax.scatter([], [], color=VERM, label="does not"); ax.scatter([], [], color=PLUM, marker="^", label="invented country")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("SmolLM2-135M, greedy: most points lie below the diagonal", fontsize=11)
save(fig, p("fig37-06-name-share.png"))

# 04 Tiny's reliability before and after one temperature (calibration.json bins at T=1; ECE on the testing half)
d = J("calibration")["Tiny"]
fig, ax = plt.subplots(figsize=(6.6, 4.4))
names = ["fitting half,\nT = 1", "testing half,\nT = 1", f"testing half,\nT = {d['temp']:.2f}"]
vals = [d["ece"], d["ece_half"], d["ece_half_scaled"]]
ax.bar(range(3), vals, color=[GREY, VERM, TEAL])
for i, v in enumerate(vals): ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels(["all 160 windows,\nT = 1", "testing half,\nT = 1", f"testing half,\nT = {d['temp']:.2f}"])
ax.set_ylabel("expected calibration error"); ax.set_title("Tiny on Through the Looking-Glass: one temperature", fontsize=11)
save(fig, p("fig37-04-temperature.png"))
print("ok", sorted(os.listdir(OUT)))
