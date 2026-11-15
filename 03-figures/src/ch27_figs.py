"""Figures for Chapter 27, Generating Text.  Every number is read from 04-code/ch27/outputs/ (run the chapter's scripts
first: 04-code/ch27/RUN.txt).  Run from the project root: python3 03-figures/src/ch27_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch27"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch27", "outputs")
J = lambda n: json.load(open(os.path.join(CH, n + ".json")))
p = lambda n: os.path.join(OUT, n)
B, E, S = J("generate_basics"), J("experiments"), J("stopping")

# 01 temperature: top share and entropy count against T
C = np.array(B["C"])
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.plot(C[:, 0], np.exp(C[:, 2]), "o-", color=BLUE, lw=2, label="equally likely choices (e^entropy)")
ax.set_xlabel("temperature T"); ax.set_ylabel("equally likely choices", color=BLUE); ax.set_yscale("log")
ax2 = ax.twinx(); ax2.plot(C[:, 0], C[:, 1], "s--", color=VERM, lw=2, label="share of the top token (line break)")
ax2.set_ylabel("share of the top token", color=VERM); ax2.grid(False)
for T, n in zip(C[:, 0], np.exp(C[:, 2])):
    ax.annotate(f"{n:.0f}", (T, n), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color=BLUE)
ax.set_title('Tiny after "the cat sat on the": one dial, sharp to flat')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
save(fig, p("fig27-01-temperature.png"))

# 02 matched settings: invented words against surprise
F = E["F"]
fig, ax = plt.subplots(figsize=(8.6, 4.8))
for (lab, r), col in zip(F.items(), SERIES):
    ax.scatter(r["surprise"], r["novel"], s=70, color=col, zorder=3)
    ax.annotate(f"{lab} (keeps {r['kept']:.0f})", (r["surprise"], r["novel"]), textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set_xlabel("mean surprise of the chosen tokens (nats, untempered model)"); ax.set_ylabel("share of words not in the pool")
ax.set_title("Seven rules, 60 continuations each: fewer kept, fewer invented")
save(fig, p("fig27-02-matched.png"))

# 03 penalties on greedy decoding
P = E["G"]["penalties"]; labs = list(P)
x = np.arange(len(labs)); w = 0.38
fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.bar(x - w / 2, [P[k]["greedy"]["rep"] for k in labs], w, color=BLUE, label="repeated 4-token runs")
ax.bar(x + w / 2, [P[k]["greedy"]["novel"] for k in labs], w, color=AMBER, label="words not in the pool")
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9); ax.set_ylabel("share")
ax.set_title("Greedy decoding, 200 tokens, three prompts: four penalties")
ax.legend(fontsize=9)
save(fig, p("fig27-03-penalties.png"))

# 04 beam width
H = E["H"]; ws = [1, 2, 4, 8, 16]
fig, ax = plt.subplots(figsize=(8.2, 4.6))
for i, (col, mk) in enumerate(zip((BLUE, VERM, TEAL), "osD")):
    ax.plot(ws, [H[str(wd)][i]["lp"] for wd in ws], marker=mk, color=col, lw=2, label=f"prompt {i + 1}")
ax.set_xscale("log", base=2); ax.set_xticks(ws); ax.set_xticklabels(ws)
ax.set_xlabel("beam width"); ax.set_ylabel("total log-probability of 30 tokens")
ax.set_title("Wider beams find likelier text, not always step by step")
ax.legend(fontsize=9)
save(fig, p("fig27-04-beam.png"))

# 05 constrained choice: exact against token-by-token masking
Jn = E["J"]; names = [n.strip() for n in Jn["names"]]
x = np.arange(len(names)); w = 0.27
fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.bar(x - w, Jn["exact"], w, color=BLUE, label="model's own chances among the six")
ax.bar(x, Jn["local"], w, color=VERM, label="token-by-token masking (exact)")
ax.bar(x + w, Jn["draws"], w, color=AMBER, label="400 masked draws")
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("chance")
ax.set_title('"Come here, my dear," said ... : masking moves the choice')
ax.legend(fontsize=9)
save(fig, p("fig27-05-masking.png"))

# 06 stopping lengths
labs = list(S)
fig, ax = plt.subplots(figsize=(9.0, 4.6))
for i, (k, col) in enumerate(zip(labs, SERIES)):
    L = np.array(S[k]["lengths"]); jit = (np.arange(len(L)) % 7 - 3) * 0.04
    ax.scatter(i + jit, L, color=col, s=22, alpha=0.8)
    ax.plot([i - 0.25, i + 0.25], [np.median(L)] * 2, color=NAVY, lw=2)
ax.axhline(200, color=GREY, ls=":"); ax.text(len(labs) - 0.5, 203, "cap", fontsize=9, color=GREY)
ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=9); ax.set_ylabel("tokens before the first blank line")
ax.set_title("Thirty draws per rule: hotter rules run longer")
save(fig, p("fig27-06-stopping.png"))
print("6 figures written to", OUT)
