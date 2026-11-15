"""Figures for Chapter 39, Fine-Tuning in Miniature.  Numbers are read from 04-code/ch39/outputs/ and
04-code/ch38/outputs/ (run 04-code/ch39/RUN.txt first).  Run from the project root: python3 03-figures/src/ch39_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch39"); os.makedirs(OUT, exist_ok=True)
O = os.path.join(ROOT, "04-code", "ch39", "outputs")
J = lambda n: json.load(open(os.path.join(O, n + ".json")))
p = lambda n: os.path.join(OUT, n)
num = lambda s: int(s.split("/")[0])

# 01 trainable numbers per method (log scale) and facts right after 600 steps
runs = [("full_check", "full"), ("lora_r1", "LoRA qv r1"), ("lora_r4", "LoRA qv r4"), ("lora_r8", "LoRA qv r8"), ("lora_r36", "LoRA qv r36"),
        ("lora_r8all", "LoRA all r8"), ("adapter", "bottleneck 16"), ("bias", "biases + norms"), ("head", "last block")]
D = [J("tune_" + n) for n, _ in runs]
fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.6))
y = np.arange(len(runs))[::-1]
ax = axs[0]; ax.barh(y, [d["trainable"] for d in D], color=[VERM] + [BLUE] * 5 + [TEAL, PLUM, AMBER]); ax.set_xscale("log")
for yi, d in zip(y, D): ax.text(d["trainable"] * 1.15, yi, f"{d['trainable']:,}", va="center", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels([l for _, l in runs], fontsize=9); ax.set_xlabel("trainable numbers (log scale)"); ax.set_xlim(3e3, 3e7)
ax.set_title("What trains", fontsize=10)
ax = axs[1]; f = [num(d["log"][-1]["exact"]["fact"]) for d in D]; pl = [d["log"][-1]["pool"] for d in D]
ax.barh(y + 0.2, f, 0.38, color=TEAL, label="facts right of 30")
ax2 = ax.twiny(); ax2.barh(y - 0.2, [x - 3.5367 for x in pl], 0.38, color=GREY, label="pool loss rise (nats)")
for yi, a, b in zip(y, f, pl): ax.text(a + 0.3, yi + 0.2, str(a), va="center", fontsize=8); ax2.text(b - 3.5367 + 0.03, yi - 0.2, f"+{b - 3.5367:.2f}", va="center", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlim(0, 34); ax2.set_xlim(0, 2.6); ax.set_xlabel("facts right of 30 (teal)"); ax2.set_xlabel("rise in own-pool loss after 600 steps (grey)")
save(fig, p("fig39-01-what-trains.png"))

# 02 learning against forgetting: facts right against pool loss at every checkpoint
fig, ax = plt.subplots(figsize=(7.8, 5.0))
for name, lab, col, mk in (("full_check", "full, 3e-4", VERM, "o"), ("lora_r8all_1800", "LoRA all r8, 1e-3", BLUE, "s"), ("lora_r36", "LoRA qv r36, 1e-3", PLUM, "^"),
                           ("adapter_1800", "bottleneck 16, 1e-3", TEAL, "D"), ("lora_r8all_replay4", "LoRA all r8 + 4 replay windows", AMBER, "v")):
    d = J("tune_" + name)["log"]
    ax.plot([m["pool"] for m in d], [num(m["exact"]["fact"]) for m in d], marker=mk, color=col, label=lab)
ax.axvline(3.5367, color=SLATE, lw=0.8, ls=":"); ax.text(3.56, 31, "before", fontsize=8, color=SLATE)
ax.set_xlabel("Tiny's loss on its own pool (nats a token): forgetting"); ax.set_ylabel("taught facts right, of 30: learning")
ax.set_ylim(-1, 32); ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.set_title("Learning against forgetting, every checkpoint", fontsize=11)
save(fig, p("fig39-02-frontier.png"))

# 03 the same 221,184 numbers in two places, over steps
fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.2))
for name, lab, col in (("lora_r36", "attention only, rank 36", PLUM), ("lora_r8all", "every matrix, rank 8", BLUE)):
    d = J("tune_" + name)["log"]; st = [m["step"] for m in d]
    axs[0].plot(st, [num(m["exact"]["fact"]) for m in d], marker="o", color=col, label=lab)
    axs[1].plot(st, [m["pool"] for m in d], marker="o", color=col, label=lab)
axs[0].set_xlabel("step"); axs[0].set_ylabel("facts right, of 30"); axs[0].legend(frameon=False, fontsize=8)
axs[1].set_xlabel("step"); axs[1].set_ylabel("own-pool loss (nats a token)")
fig.suptitle("221,184 trainable numbers each: where they go matters", fontsize=11)
save(fig, p("fig39-03-where.png"))
print("ok", sorted(os.listdir(OUT)))
