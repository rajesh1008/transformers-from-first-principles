"""Figures for Chapter 40, Interpretability: Probes and Attention Maps.  Numbers are read from 04-code/ch40/outputs/ (run 04-code/ch40/RUN.txt first);
figure 3 runs Chapter 38's fine-tuned Tiny on one prompt.  Run from the project root: python3 03-figures/src/ch40_figs.py"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from style import *
import matplotlib.pyplot as plt
import numpy as np
setup()
OUT = os.path.join(ROOT, "03-figures", "ch40"); os.makedirs(OUT, exist_ok=True)
O = os.path.join(ROOT, "04-code", "ch40", "outputs")
J = lambda n: json.load(open(os.path.join(O, n + ".json")))
p = lambda n: os.path.join(OUT, n)
L = list(range(13))

# 01 probes by layer, trained against untrained
R = J("probe")
fig, axs = plt.subplots(1, 4, figsize=(14.0, 3.9))
def panel(ax, task, ctl, title):
    ax.plot(L, R["trained"][task], color=BLUE, marker="o", ms=3.5, label="trained Tiny")
    ax.plot(L, R["untrained"][task], color=AMBER, marker="s", ms=3.5, label="untrained Tiny")
    if ctl: ax.plot(L, R["trained"][ctl], color=VERM, ls="--", label="control task, trained")
    ax.axhline(R["majority"][task], color=GREY, lw=1, ls=":"); ax.text(0.2, R["majority"][task] + 0.008, "majority", fontsize=7, color=GREY)
    ax.set_title(title, fontsize=9.5); ax.set_xlabel("layer"); ax.set_xticks([0, 3, 6, 9, 12])
panel(axs[0], "SPACE", "CTL_SPACE", "current token starts a word"); axs[0].set_ylabel("held-out accuracy"); axs[0].legend(fontsize=7, loc="center right")
panel(axs[1], "NEXTSPACE", "CTL_NEXTSPACE", "next token starts a word")
panel(axs[2], "PREV", None, "which common token came before")
panel(axs[3], "POS", None, "quarter of the window")
fig.suptitle("What a linear probe reads from each layer's rows", fontsize=11, x=0.01, ha="left")
save(fig, p("fig40-01-probes.png"))

# 02 the taught fact: probe by layer and patch by layer
F = J("facts"); F2 = J("facts2")
fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.2))
ax = axs[0]
ax.plot(L, F["probe_site_fine-tuned"], color=BLUE, marker="o", label="fine-tuned, at the answer's animal name")
ax.plot(L, F["probe_site_base"], color=AMBER, marker="s", label="base, same seat")
ax.plot(L, F["probe_question_fine-tuned"], color=PLUM, marker="^", ls="--", label='fine-tuned, at the T of "Tiny:"')
ax.axhline(0.233, color=GREY, ls=":"); ax.text(0.2, 0.25, "majority verb", fontsize=8, color=GREY)
ax.set_xlabel("layer"); ax.set_ylabel("leave-one-animal-out accuracy, 8 verbs"); ax.set_ylim(0, 1.03); ax.legend(fontsize=7.5, loc="upper left"); ax.set_title("A probe for the verb", fontsize=10)
ax = axs[1]; pt = F2["patch"]
ax.plot(L, [a for a, b in pt], color=TEAL, marker="o", label="prediction becomes the donor's verb")
ax.plot(L, [b for a, b in pt], color=VERM, marker="s", label="stays the recipient's verb")
ax.set_xlabel("layer at which the site's row is replaced"); ax.set_ylabel("share of 200 pairs"); ax.set_ylim(-0.02, 1.05); ax.legend(fontsize=7.5, loc="center right"); ax.set_title("A patch of the same row", fontsize=10)
save(fig, p("fig40-02-fact.png"))

# 03 attention maps on one prompt: four heads, fine-tuned Tiny
sys.path.insert(0, os.path.join(ROOT, "04-code", "ch40"))
os.chdir(os.path.join(ROOT, "04-code", "ch40"))
from lib import *
T = tok(); ft = load_ft()
ids = T.encode("User: What did the owl do?\nTiny:") + T.encode(" The") + T.encode(" owl")
labels = [T.decode([i]).replace("\n", "↵").replace(" ", "␣") for i in ids]
with torch.no_grad(): A = forward(ft, ids, want_att=True)["att"]
heads = [(0, 1), (5, 3), (7, 0), (11, 2)]
fig, axs = plt.subplots(1, 4, figsize=(14.2, 4.4))
for ax, (l, h) in zip(axs, heads):
    m = A[l][0, h].numpy(); ax.imshow(m, cmap="Blues", vmin=0, vmax=max(0.5, m.max()), aspect="equal")
    ax.set_xticks(range(len(ids))); ax.set_xticklabels(labels, rotation=90, fontsize=6.5); ax.set_yticks(range(len(ids))); ax.set_yticklabels(labels, fontsize=6.5)
    ax.grid(False); ax.set_title(f"head {l}.{h}", fontsize=10); ax.spines[:].set_visible(False)
axs[0].set_ylabel("query seat (reads)"); fig.supxlabel("key seat (read from)", fontsize=9)
fig.suptitle("Share tables of four heads, fine-tuned Tiny, on one prompt", fontsize=11, x=0.01, ha="left")
save(fig, p("fig40-03-maps.png"))

# 04 attention against ablation, and removing k heads
H = J("facts"); F2 = J("facts2")
fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.3))
ax = axs[0]; att = np.array(H["att_share"]).ravel(); ab = np.array(H["ablation"]).ravel()
ax.scatter(att, ab, color=BLUE, s=22, alpha=0.85)
for i in np.argsort(-ab)[:3]: ax.annotate(f"{i // 4}.{i % 4}", (att[i], ab[i]), fontsize=8, xytext=(4, 2), textcoords="offset points")
for i in np.argsort(-att)[:2]: ax.annotate(f"{i // 4}.{i % 4}", (att[i], ab[i]), fontsize=8, xytext=(4, 2), textcoords="offset points")
ax.set_xlabel("attention share on the animal name in the question"); ax.set_ylabel("drop in log p(verb) when the head is zeroed")
ax.set_title(f"48 heads: Spearman {H['rho']:.2f}", fontsize=10)
ax = axs[1]; K = [1, 4, 8, 16, 32, 48]; kh = F2["khead"]
for nm, col, mk in (("by attention", PLUM, "o"), ("by ablation", VERM, "s"), ("random (mean of 5)", GREY, "^")):
    ax.plot(K, [kh[str(k)][nm][0] for k in K], color=col, marker=mk, label=f"top k {nm}" if nm != "random (mean of 5)" else "k random heads (mean of 5)")
ax.set_xlabel("heads zeroed, k"); ax.set_ylabel("drop in mean log p(verb)"); ax.legend(fontsize=8); ax.set_title("Removing the heads attention points at", fontsize=10)
save(fig, p("fig40-04-faith.png"))
print("ok", sorted(os.listdir(OUT)))
