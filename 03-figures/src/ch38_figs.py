"""Figures for Chapter 38, From Predictor to Assistant.  Numbers are read from 04-code/ch38/outputs/ (run
04-code/ch38/RUN.txt first).  Run from the project root: python3 03-figures/src/ch38_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch38"); os.makedirs(OUT, exist_ok=True)
O = os.path.join(ROOT, "04-code", "ch38", "outputs")
J = lambda n: json.load(open(os.path.join(O, n + ".json")))
p = lambda n: os.path.join(OUT, n)
num = lambda s: int(s.split("/")[0])

# 01 facts learnt by how often they were taught, against steps (sft_response)
d = J("sft_response")["log"]; st = [m["step"] for m in d]
fig, ax = plt.subplots(figsize=(7.6, 4.6))
for k, col, mk in (("fact x12", TEAL, "o"), ("fact x4", BLUE, "s"), ("fact x1", VERM, "D")):
    ax.plot(st, [num(m["exact"][k]) for m in d], marker=mk, color=col, label=f"taught {k.split('x')[1]} time{'s' if k != 'fact x1' else ''}")
ax.plot(st, [num(m["exact"]["fact-reworded"]) / 3 for m in d], marker="^", color=PLUM, ls="--", label="same facts reworded (of 30, scaled to 10)")
ax.set_xlabel("fine-tuning step (16 pairs a step)"); ax.set_ylabel("right, of 10"); ax.set_ylim(-0.3, 10.5)
ax.legend(frameon=False, fontsize=9, loc="center right"); ax.set_title("Tiny learns the facts it was shown most often first", fontsize=11)
save(fig, p("fig38-01-facts-by-repetition.png"))

# 02 forgetting: pool and Looking-Glass loss against steps, with and without replay (sft_response, sft_response_replay4)
fig, ax = plt.subplots(figsize=(7.6, 4.6))
for name, lab, col in (("sft_response", "reply only", VERM), ("sft_response_replay4", "reply + 4 pool windows a step", TEAL)):
    d = J(name)["log"]; st = [m["step"] for m in d]
    ax.plot(st, [m["pool"] for m in d], color=col, marker="o", label=f"{lab}: own pool")
    ax.plot(st, [m["glass"] for m in d], color=col, marker="s", ls="--", label=f"{lab}: Looking-Glass")
ax.set_xlabel("fine-tuning step"); ax.set_ylabel("ordinary loss, nats a token")
ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.set_title("Forgetting, and what replay keeps", fontsize=11)
save(fig, p("fig38-02-forgetting.png"))

# 03 preference tuning: first reply token and outcomes (dpo_v2)
d = J("dpo_v2")
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.2))
ax = axs[0]; x = np.arange(2)
ax.bar(x - 0.18, [d["before"]["first_The"], d["after"]["first_The"]], 0.34, color=BLUE, label="' The'")
ax.bar(x + 0.18, [d["before"]["first_I"], d["after"]["first_I"]], 0.34, color=AMBER, label="' I'")
ax.set_xticks(x); ax.set_xticklabels(["before", "after 300 steps"]); ax.set_ylabel("share as first reply token\n(mean over 30 taught questions)")
ax.legend(frameon=False, fontsize=9); ax.set_ylim(0, 1.1)
ax = axs[1]
B, A = d["before"], d["after"]
labels = ["taught facts\nright (of 30)", "taught facts\n'I do not know.'", "unknown, held out\n'I do not know.' (of 10)"]
bv = [B["x1"]["right"] + B["x4"]["right"] + B["x12"]["right"], B["x1"]["idk"] + B["x4"]["idk"] + B["x12"]["idk"], B["unknown_test"]["idk"]]
av = [A["x1"]["right"] + A["x4"]["right"] + A["x12"]["right"], A["x1"]["idk"] + A["x4"]["idk"] + A["x12"]["idk"], A["unknown_test"]["idk"]]
x = np.arange(3); ax.bar(x - 0.18, bv, 0.34, color=GREY, label="before"); ax.bar(x + 0.18, av, 0.34, color=VERM, label="after")
for i in range(3): ax.text(i - 0.18, bv[i] + 0.4, str(bv[i]), ha="center", fontsize=9); ax.text(i + 0.18, av[i] + 0.4, str(av[i]), ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.legend(frameon=False, fontsize=9)
fig.suptitle("Preference tuning on 50 pairs, lr 1e-4, beta 0.1", fontsize=11)
save(fig, p("fig38-03-preference.png"))
print("ok", sorted(os.listdir(OUT)))

# 04 (tick 30) four preference runs: outcomes and the chosen reply's log-probability change (dpo_v3, dpo_knownidk, dpo_beta0.5, dpo_sftw1)
runs = [("dpo_v3", "beta 0.1"), ("dpo_knownidk", "+ pairs rejecting\n'I do not know.'"), ("dpo_beta0.5", "beta 0.5"), ("dpo_sftw1", "+ chosen reply's\nordinary loss")]
R4 = [J(n) for n, _ in runs]
fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.4), gridspec_kw=dict(width_ratios=[1.7, 1]))
ax = axs[0]; x = np.arange(len(runs)); w = 0.26
vals = [[r["after"]["x1"]["right"] + r["after"]["x4"]["right"] + r["after"]["x12"]["right"] for r in R4],
        [r["after"]["unknown_train_idk"] for r in R4], [r["after"]["unknown_test"]["idk"] for r in R4]]
for j, (v, lab, col) in enumerate(zip(vals, ["taught facts right (of 30)", "trained unknown declined (of 20)", "held-out unknown declined (of 10)"], [TEAL, BLUE, AMBER])):
    ax.bar(x + (j - 1) * w, v, w, color=col, label=lab)
    for i, t in enumerate(v): ax.text(i + (j - 1) * w, t + 0.4, str(t), ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in runs], fontsize=8); ax.set_ylim(0, 34); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("After 300 steps of preference tuning", fontsize=10)
ax = axs[1]; v = [r["logp_change"]["known"]["chosen"] for r in R4]
ax.bar(x, v, 0.55, color=[VERM if t < -1 else TEAL for t in v]); ax.axhline(0, color=SLATE, lw=0.8)
for i, t in enumerate(v): ax.text(i, t - 2.2 if t < 0 else t + 0.8, f"{t:+.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(["beta 0.1", "+ reject IDK", "beta 0.5", "+ ordinary loss"], fontsize=8)
ax.set_ylabel("change in log p of the chosen reply,\nknown facts (nats)"); ax.set_ylim(-36, 6)
ax.set_title("The chosen reply pushed down, or kept", fontsize=10)
save(fig, p("fig38-04-repairs.png"))

# 05 (tick 30) the running sentence before and after fine-tuning (shares)
S = J("shares"); names = ["base", "sft_3e-4_600.pt", "sft_1e-3_4000.pt"]; labs = ["base Tiny", "600 steps", "4,000 steps"]
fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axs[0]; x = np.arange(2); w = 0.26
for j, (n, lab, col) in enumerate(zip(names, labs, [GREY, BLUE, VERM])):
    v = [S[n]["plain"]["mat"], S[n]["template"]["mat"]]; ax.bar(x + (j - 1) * w, v, w, color=col, label=lab)
    for i, t in enumerate(v): ax.text(i + (j - 1) * w, t + 0.02, f"{t:.4f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(["'the cat sat on the'", "inside the template, after\n'Tiny: The cat sat on the'"], fontsize=8)
ax.set_ylabel("share of ' m' (first token of ' mat')"); ax.set_ylim(0, 1.12); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax = axs[1]; v = [S[n]["plain"]["eot"] for n in names]
ax.bar(np.arange(3), v, 0.55, color=[GREY, BLUE, VERM])
for i, t in enumerate(v): ax.text(i, t + 0.004, f"{t:.4f}", ha="center", fontsize=8)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(labs, fontsize=8); ax.set_ylabel("end-marker share after\n'the cat sat on the'"); ax.set_ylim(0, 0.16)
fig.suptitle("The running sentence inside and outside the template", fontsize=11)
save(fig, p("fig38-05-running.png"))
print("ok tick 30", sorted(os.listdir(OUT)))
