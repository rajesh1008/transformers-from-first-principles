"""Figures for Chapter 25, Evaluating a Language Model.  Every number is read from 04-code/ch25/outputs/ (run the
chapter's scripts first: 04-code/ch25/RUN.txt).  Run from the project root: python3 03-figures/src/ch25_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch25"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch25", "outputs")
J = lambda n: json.load(open(os.path.join(CH, n + ".json")))
p = lambda n: os.path.join(OUT, n)
EV = J("evaluate")

# 01 the running sentence, surprise by token
s = EV["sentence"]
fig, ax = plt.subplots(figsize=(8.4, 4.4))
labs = ["the → cat", "the cat → sat", "... sat → on", "... on → the", "... the → m", "... m → at"]
ax.bar(range(6), s["surprise"], color=BLUE)
for i, (x, pr) in enumerate(zip(s["surprise"], s["p"])):
    ax.text(i, x + 0.15, f"p {pr:.4f}\n{x:.2f}", ha="center", fontsize=9)
ax.axhline(s["loss"], ls="--", color=VERM); ax.text(5.45, s["loss"] + 0.15, f"average {s['loss']:.2f}\nperplexity {s['ppl']:.0f}", color=VERM, fontsize=9, ha="right")
ax.set_xticks(range(6)); ax.set_xticklabels(labs, fontsize=9, rotation=15)
ax.set_ylim(0, 10); ax.set_ylabel("surprise, −ln p (nats)")
ax.set_title("The running sentence under the 2,500-step Tiny")
save(fig, p("fig25-01-sentence.png"))

# 02 frequency bands, Looking-Glass
b = [r for r in EV["bands"]["glass"] if r.get("n")]
fig, ax = plt.subplots(figsize=(8.6, 4.8))
x = np.arange(len(b)); w = 0.26
ax.bar(x - w, [r["tiny2500"] for r in b], w, color=BLUE, label="Tiny, 2,500 steps")
ax.bar(x, [r["bigram"] for r in b], w, color=AMBER, hatch="//", label="bigram counts")
ax.bar(x + w, [r["unigram"] for r in b], w, color=GREY, hatch="..", label="unigram counts")
for i, r in enumerate(b):
    ax.text(i, max(r["tiny2500"], r["bigram"], r["unigram"]) + 0.4, f"{100 * r['share']:.1f}% of tokens\n{100 * r['loss_share']:.0f}% of loss", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([r["band"] for r in b])
ax.set_xlabel("how often the true next token occurs in the Carroll pool"); ax.set_ylabel("average surprise (nats)")
ax.set_ylim(0, 18); ax.legend(loc="upper right")
ax.set_title("Looking-Glass: where the loss lives")
save(fig, p("fig25-02-bands.png"))

# 03 reliability diagram, top-1
fig, ax = plt.subplots(figsize=(6.4, 6.0))
ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
for key, col, mk, lab in (("2500_carroll", BLUE, "o", "2,500 steps, Carroll held-out"), ("2500_pool", TEAL, "s", "2,500 steps, pool text (read)"),
                          ("1250_carroll", AMBER, "D", "1,250 steps, Carroll held-out")):
    rel = [r for r in EV["calib"][key]["top1"] if r[2] >= 20]
    ax.plot([r[3] for r in rel], [r[4] for r in rel], mk + "-", color=col, label=f"{lab} (ECE {EV['calib'][key]['ece_top1']:.3f})")
ax.set_xlabel("average top-1 probability in the bin"); ax.set_ylabel("how often the top token was right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("Does 0.3 mean 30 per cent?")
save(fig, p("fig25-03-reliability.png"))

# 04 held-out draws
d = np.array(EV["draws"]["values"]); bc = EV["boot_carroll"]
fig, ax = plt.subplots(figsize=(8.2, 4.6))
ax.hist(d, bins=24, color=BLUE, alpha=1.0, edgecolor="white")
ax.axvline(bc["window"]["mean"], color=NAVY, lw=2); ax.text(bc["window"]["mean"] + 0.01, 21, f"all 31 windows {bc['window']['mean']:.4f}", fontsize=9, color=NAVY)
ax.plot(bc["naive"]["low"], 2, "|", ms=1)
ax.hlines(-1.2, bc["naive"]["low"], bc["naive"]["high"], color=VERM, lw=3); ax.text(bc["naive"]["high"] + 0.01, -1.2, "naive 95% (predictions independent)", va="center", fontsize=8.5, color=VERM)
ax.hlines(-2.8, bc["window"]["low"], bc["window"]["high"], color=TEAL, lw=3); ax.text(bc["window"]["high"] + 0.01, -2.8, "95% from resampling whole windows", va="center", fontsize=8.5, color=TEAL)
ax.set_ylim(-4, 26); ax.set_xlabel("held-out loss of the 2,500-step model on 16 windows drawn at random")
ax.set_ylabel("draws (of 200)")
ax.set_title("One held-out number, 200 ways")
save(fig, p("fig25-04-draws.png"))

# 05 paired against unpaired intervals
C = J("compare")["pairs"]
rows = [("wd0.1 2,500 | 1e-3 2,500 | glass", "AdamW 0.1 minus Adam, order 1865"), ("wd0.1 o7 2,500 | o7 2,500 | glass", "AdamW 0.1 minus Adam, order 7"),
        ("wd0.1 o11 2,500 | o11 2,500 | glass", "AdamW 0.1 minus Adam, order 11"), ("o7 2,500 | 1e-3 2,500 | glass", "order 7 minus order 1865, same recipe"),
        ("o11 2,500 | 1e-3 2,500 | glass", "order 11 minus order 1865, same recipe"), ("1e-3 2,500 | 2e-3 2,500 | glass", "lr 1e-3 minus lr 2e-3")]
fig, ax = plt.subplots(figsize=(8.8, 4.8))
for i, (k, lab) in enumerate(rows):
    r = C[k]; y = len(rows) - 1 - i
    ax.hlines(y + 0.15, *r["unpaired"], color=GREY, lw=2.2); ax.hlines(y - 0.15, *r["paired"], color=BLUE, lw=3.2)
    ax.plot(r["diff"], y, "o", color=NAVY)
    ax.text(0.105, y, lab, va="center", fontsize=9)
ax.axvline(0, color=SLATE, lw=1)
ax.plot([], [], color=GREY, lw=2.2, label="unpaired 95%"); ax.plot([], [], color=BLUE, lw=3.2, label="paired 95% (same windows)")
ax.set_yticks([]); ax.set_xlim(-0.25, 0.32); ax.set_xlabel("difference in Looking-Glass loss (nats; negative favours the first)")
ax.legend(loc="upper right", fontsize=8.5); ax.set_ylim(-0.6, 6.0); ax.set_title("Pairing narrows the interval; it does not see training noise")
save(fig, p("fig25-05-paired.png"))

# 06 contamination
K = J("contaminate")
fig, ax = plt.subplots(figsize=(8.0, 4.6))
kinds = ["exact x1", "exact x4", "altered x4"]
x = np.arange(3); w = 0.36
pl = [K[k]["planted"] - K["control"]["planted"] for k in kinds]; ot = [K[k]["other"] - K["control"]["other"] for k in kinds]
ax.bar(x - w / 2, pl, w, color=VERM, label="the 8 planted windows")
ax.bar(x + w / 2, ot, w, color=GREY, hatch="//", label="the 23 others")
ng = [K["ngram_13_pool + exact copies"]["planted"], K["ngram_13_pool + exact copies"]["planted"], K["ngram_13_pool + altered copies"]["planted"]]
for i in range(3):
    ax.text(i, pl[i] - 0.012, f"{pl[i]:+.3f}", ha="center", va="top", fontsize=9)
    ax.text(i, 0.012, f"13-gram check finds\n{100 * ng[i]:.0f}% of their 13-grams", ha="center", fontsize=8.5)
ax.axhline(0, color=SLATE, lw=1)
ax.set_xticks(x); ax.set_xticklabels(["each copy read once", "read four times", "altered, read four times"])
ax.set_ylim(-0.2, 0.06); ax.set_ylabel("change in held-out loss against the control run")
ax.legend(loc="lower left"); ax.set_title("Planted contamination, and the check that misses it")
save(fig, p("fig25-06-contamination.png"))

# 07 cloze accuracy against loss
Bn = J("bench")["results"]
fig, ax = plt.subplots(figsize=(7.6, 5.0))
for k, r in Bn.items():
    if k in ("unigram", "bigram"):
        continue
    col = AMBER if "wd" in k else BLUE; mk = "s" if "wd" in k else "o"
    if not k.startswith("2,500"):
        col, mk = GREY, "^"
    ax.errorbar(r["glass_loss"], r["glass_sum"], yerr=[[r["glass_sum"] - r["ci"][0]], [r["ci"][1] - r["glass_sum"]]], fmt=mk, color=col, capsize=3)
    if not k.startswith("2,500"):
        ax.text(r["glass_loss"] - 0.012, r["glass_sum"] + 0.006, k, fontsize=8, ha="right")
ax.axhline(0.25, ls=":", color=GREY); ax.text(ax.get_xlim()[0] if False else 4.32, 0.255, "chance", fontsize=8, color=GREY)
ax.axhline(Bn["bigram"]["glass_sum"], ls="--", color=VERM); ax.text(4.32, Bn["bigram"]["glass_sum"] + 0.005, f"bigram counts {Bn['bigram']['glass_sum']:.2f}", fontsize=8, color=VERM)
ax.errorbar([], [], fmt="o", color=BLUE, label="2,500 steps, no decay (orders 1865, 7, 11)")
ax.errorbar([], [], fmt="s", color=AMBER, label="2,500 steps, AdamW 0.1 (orders 1865, 7, 11)")
ax.errorbar([], [], fmt="^", color=GREY, label="shorter branches of one run")
ax.legend(loc="center left", fontsize=8); ax.set_xlim(4.2, 5.05)
ax.set_xlabel("Looking-Glass loss, first 128 windows (nats)"); ax.set_ylabel("cloze accuracy, 400 Looking-Glass items")
ax.set_title("A constructed benchmark against the loss")
save(fig, p("fig25-07-cloze.png"))
print("figures written to", OUT)
