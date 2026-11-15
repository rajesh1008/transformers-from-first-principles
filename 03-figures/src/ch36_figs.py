"""Figures for Chapter 36, In-Context Learning and Retrieval, Mechanically.  Numbers are read from 04-code/ch36/outputs/
(run 04-code/ch36/RUN.txt first).  Run from the project root: python3 03-figures/src/ch36_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch36"); os.makedirs(OUT, exist_ok=True)
O = os.path.join(ROOT, "04-code", "ch36", "outputs")
J = lambda n: json.load(open(os.path.join(O, n + ".json")))
p = lambda n: os.path.join(OUT, n)

# 01 induction and previous-token scores per head, GPT-2 small against Tiny
d = J("induction")
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
for ax, key, title in ((axes[0], "gpt2", "GPT-2 small (12 layers x 12 heads)"), (axes[1], "tiny", "Tiny (12 layers x 4 heads)")):
    ind = np.array(d[key]["ind"]).flatten(); prev = np.array(d[key]["prev"]).flatten()
    ax.scatter(prev, ind, s=26, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axhline(d[key]["chance"], color=GREY, ls=":", lw=1); ax.axvline(d[key]["chance"], color=GREY, ls=":", lw=1)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("previous-token score"); ax.set_title(title, fontsize=11)
    if key == "gpt2":
        I = np.array(d[key]["ind"]); P = np.array(d[key]["prev"])
        for (l, h) in ((5, 1), (5, 5), (6, 9), (7, 10)):
            ax.annotate(f"{l}.{h}", (P[l][h], I[l][h]), xytext=(6, -3), textcoords="offset points", fontsize=9, color=NAVY)
        ax.annotate("4.11", (P[4][11], I[4][11]), xytext=(-24, 8), textcoords="offset points", fontsize=9, color=NAVY)
axes[0].set_ylabel("induction score (share on the token after the earlier copy)")
axes[1].annotate("every head within 0.15 of the\neven-spread line on both axes", (0.25, 0.5), fontsize=10, color=SLATE)
fig.suptitle("Random tokens shown twice: where each head looks on the second showing", fontsize=12)
save(fig, p("fig36-01-induction-heads.png"))

# 02 continued training: induction score and repeat loss by step
runs = [("continue_plain_long", GREY, "ordinary windows (control)", "o"), ("continue_repeat_long64", BLUE, "64-token passage shown twice", "s"),
        ("continue_repeat_long32", VERM, "32-token passage shown four times", "D")]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
for name, col, lab, mk in runs:
    if not os.path.exists(os.path.join(O, name + ".json")): continue
    L = J(name)["log"]; st = [r["step"] for r in L]
    axes[0].plot(st, [r["ind"] for r in L], marker=mk, color=col, label=lab, ms=4)
    axes[1].plot(st, [r["repeat"] for r in L], marker=mk, color=col, ms=4, label=lab + ", repeat")
    axes[1].plot(st, [r["held"] for r in L], marker=mk, color=col, ms=4, ls="--", alpha=0.6)
axes[0].set_xlabel("extra training steps (AdamW, lr 1e-3)"); axes[0].set_ylabel("best induction score over 48 heads")
axes[1].set_xlabel("extra training steps"); axes[1].set_ylabel("nats per token")
axes[1].set_title("solid: second showing of Looking-Glass passages\ndashed: ordinary held-out windows", fontsize=10)
axes[0].legend(frameon=False, fontsize=9, loc="center right")
save(fig, p("fig36-02-teaching-tiny.png"))

# 03 few-shot: accuracy and log p by number of examples
d = J("fewshot"); ks = [0, 1, 2, 4, 8]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
for name, col, lab, mk in (("gpt2", VERM, "GPT-2 small", "o"), ("smollm2", BLUE, "SmolLM2-135M", "s")):
    axes[0].plot(ks, [d[name][f"arrow k={k}"]["acc"] * 24 for k in ks], marker=mk, color=col, label=lab)
    axes[1].plot(ks, [d[name][f"arrow k={k}"]["lp"] for k in ks], marker=mk, color=col, label=lab)
axes[0].set_xlabel("worked examples in the prompt"); axes[0].set_ylabel("right first token, of 24 queries")
axes[1].set_xlabel("worked examples in the prompt"); axes[1].set_ylabel("mean log-probability of the right answer")
axes[0].legend(frameon=False); axes[0].set_xticks(ks); axes[1].set_xticks(ks)
fig.suptitle('Antonyms, "hot -> cold": what each extra example buys', fontsize=12)
save(fig, p("fig36-03-fewshot-k.png"))

# 04 format and labels at k = 4
tags = [("arrow k=4", "arrow"), ("io k=4", "Input/Output"), ("space k=4", "bare space"), ("mixed k=4", "mixed separators"), ("arrow k=4 shuffled", "answers shuffled")]
fig, ax = plt.subplots(figsize=(8.4, 4.6)); x = np.arange(len(tags)); w = 0.38
for i, (name, col, lab) in enumerate((("gpt2", VERM, "GPT-2 small"), ("smollm2", BLUE, "SmolLM2-135M"))):
    v = [d[name][t]["acc"] * 24 for t, _ in tags]
    b = ax.bar(x + (i - 0.5) * w, v, w, color=col, label=lab)
    for xx, vv in zip(x + (i - 0.5) * w, v): ax.text(xx, vv + 0.3, f"{vv:.0f}", ha="center", fontsize=9, color=SLATE)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in tags]); ax.set_ylabel("right first token, of 24 queries")
ax.set_title("Four examples, five ways of writing them", fontsize=12); ax.legend(frameon=False)
save(fig, p("fig36-04-format.png"))

# 05 notes: agree, contradict, among five
d = J("docs")
fig, ax = plt.subplots(figsize=(8.8, 4.6))
cols = [("A", "closed book: true city"), ("Bagree", "agreeing note: follows"), ("Bcon", "contradicting note: follows"),
        ("C1", "among five, first: follows"), ("C5", "among five, last: follows")]
def val(r, c):
    for k, v in r.items():
        if c == "A" and k.startswith("A "): return sum(v["acc"])
        if c == "Bagree" and k.startswith("B note agrees"): return sum(v["follow"])
        if c == "Bcon" and k.startswith("B note contradicts"): return sum(v["follow"])
        if c == "C1" and "position 1 of 5" in k: return sum(v["follow"])
        if c == "C5" and "position 5 of 5" in k: return sum(v["follow"])
x = np.arange(len(cols)); w = 0.27
for i, (name, col, lab) in enumerate((("gpt2", VERM, "GPT-2 small"), ("smollm2", BLUE, "SmolLM2-135M"), ("smollm2i", TEAL, "SmolLM2-135M-Instruct"))):
    v = [val(d[name], c) for c, _ in cols]
    ax.bar(x + (i - 1) * w, v, w, color=col, label=lab)
    for xx, vv in zip(x + (i - 1) * w, v): ax.text(xx, vv + 0.2, str(vv), ha="center", fontsize=9, color=SLATE)
ax.set_xticks(x); ax.set_xticklabels([l.replace(": ", ":\n") for _, l in cols], fontsize=9); ax.set_ylabel("of 12 capital-city questions")
ax.set_title("A pasted note against what the model already knows", fontsize=12); ax.legend(frameon=False, fontsize=9, loc="upper right")
save(fig, p("fig36-05-notes.png"))
print("ok", sorted(os.listdir(OUT)))
