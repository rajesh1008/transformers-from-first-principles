"""Figures for Chapter 14, Stacking Layers.  Every number is read from 04-code/ch14/outputs (the .txt and .json
files the chapter's scripts wrote) or recomputed here from tiny/model.py with seed 1865.
Run from the project root: python3 03-figures/src/ch14_figs.py"""
import os, sys, re, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code"))
from style import *
import matplotlib.pyplot as plt
setup()
OUT = os.path.join(ROOT, "03-figures", "ch14"); os.makedirs(OUT, exist_ok=True)
CH = os.path.join(ROOT, "04-code", "ch14", "outputs")
def txt(name): return open(os.path.join(CH, name + ".txt")).read()
def js(name): return json.load(open(os.path.join(CH, name + ".json")))

# 01 Signature 4: one layer, exploded (lengths of ' sat' after each step, from exploded.txt)
lens = [float(x) for x in re.findall(r"\|\s+([\d.]+) \| \[", txt("exploded"))]
fig, ax = canvas(7.2, 10.2)
steps = [("row in", AMBER, "x"), ("norm (ln_1)", TEAL, "256 numbers"), ("attention", NAVY, "65,536 numbers"),
         ("add", PLUM, "x + attention"), ("norm (ln_2)", TEAL, "256 numbers"), ("feed-forward", NAVY, "131,712 numbers"),
         ("add: row out", PLUM, "+ feed-forward")]
ys = [4, 17, 30, 43, 56, 69, 84]
for (name, c, sub), y, L in zip(steps, ys, lens):
    box(ax, 14, y, 30, 8, name, color=c, fs=11, sub=sub)
    label(ax, 48, y + 4, f"' sat' length {L:.3f}", color=SLATE, fs=10, ha="left")
for a, b in zip(ys[:-1], ys[1:]):
    arrow(ax, 29, a + 8, 29, b)
arrow(ax, 13, 8, 13, 46, color=PLUM, lw=2.5, rad=0.35); arrow(ax, 13, 47, 13, 87, color=PLUM, lw=2.5, rad=0.35)
label(ax, 3.5, 27, "kept row", color=PLUM, fs=9.5, rotation=90); label(ax, 3.5, 66, "kept row", color=PLUM, fs=9.5, rotation=90)
label(ax, 36, 99, "Tiny's layer 0 on the running sentence (7 rows of 128)", color=NAVY, fs=11, weight="bold")
label(ax, 36, 95.5, "only the attention step lets rows read each other", color=GREY, fs=9.5)
save(fig, os.path.join(OUT, "fig14-01-layer-exploded.png"))

# 02 Signature 5: the stack, with ' sat''s length entering each layer
sat = js("stack_sat")
fig, ax = canvas(7.2, 10.4)
box(ax, 10, 2, 40, 6, "token table + position table", color=AMBER, fs=10)
for l in range(12):
    y = 11 + l * 6.6
    box(ax, 14, y, 32, 5, f"layer {l}", color=NAVY, fs=9.5, sub=None)
    label(ax, 49, y + 2.5, f"' sat' enters at {sat['sat_length'][l]:.2f}", color=SLATE, fs=8.8, ha="left")
y_top = 11 + 12 * 6.6
box(ax, 14, y_top, 32, 5, "final norm (ln_f)", color=TEAL, fs=9.5)
label(ax, 49, y_top + 2.5, f"leaves layer 11 at {sat['sat_length'][12]:.2f}; ln_f gives 11.31", color=SLATE, fs=8.8, ha="left")
box(ax, 10, y_top + 7.5, 40, 6, "scores: agreement with every token row", color=PLUM, fs=9.5)
arrow(ax, 30, 8, 30, 11); arrow(ax, 30, y_top - 1.6, 30, y_top); arrow(ax, 30, y_top + 5, 30, y_top + 7.5)
label(ax, 30, y_top + 16.5, "Tiny: 2,766,592 learned numbers", color=NAVY, fs=11, weight="bold")
label(ax, 4, 50, "the running sentence flows up", color=GREY, fs=9.5, rotation=90)
save(fig, os.path.join(OUT, "fig14-02-stack.png"))

# 03 ' sat''s row: cosine with the row below and with the starting row, against the accumulation rules
st = txt("stack")
lines = [ln.split("|") for ln in st.splitlines()[2:15]]
below = [float(c[4]) for c in lines[1:]]; start = [float(c[5]) for c in lines]
fig, ax = plt.subplots(figsize=(7.2, 6.0))
xs = np.arange(1, 13)
ax.plot(xs, below, "o-", color=NAVY, label="' sat': cosine with the row one layer below")
ax.plot(range(0, 13), start, "s-", color=AMBER, label="' sat': cosine with the row entering layer 0")
ref = np.arange(1, 25)
ax.plot(ref[1::2] / 2, np.sqrt(ref[1::2] / (ref[1::2] + 1)), ":", color=GREY, label="equal unrelated writes, two per layer:\ncosine with the row below")
ax.plot(np.arange(0, 13), 1 / np.sqrt(2 * np.arange(0, 13) + 1), "--", color=GREY, label="the same writes: cosine with the start")
ax.set_xlabel("layers passed"); ax.set_ylabel("cosine"); ax.set_ylim(0, 1.02); ax.set_xticks(range(0, 13))
ax.legend(fontsize=8.8, loc="center right"); ax.set_title("Neighbouring layers look alike because the stream is a sum")
save(fig, os.path.join(OUT, "fig14-03-sat-cosines.png"))

# 04 untrained Tiny's Logit Lens: share of seats whose top token is the seat's own token
s = js("stack")
fig, ax = plt.subplots(figsize=(7.2, 5.4))
ax.plot(range(13), s["own"], "o-", color=PLUM, label="top token is the seat's own token")
ax.plot(range(13), s["sim"], "s--", color=VERM, label="token similarity of the 1,024 rows")
ax.set_xticks(range(13)); ax.set_xticklabels([str(i) for i in range(12)] + ["top"])
ax.set_xlabel("row entering layer"); ax.set_ylabel("share of seats / similarity"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=9.5); ax.set_title("Untrained Tiny: the Logit Lens reads the input token, then noise")
save(fig, os.path.join(OUT, "fig14-04-tiny-lens.png"))

# 05 where the learned numbers are: Tiny and GPT-2 small
p = txt("params")
parts_t = [("token table", 262144), ("position table", 131072), ("attention (12 layers)", 786432),
           ("feed-forward (12 layers)", 1580544), ("norms", 6144 + 256)]
g_layer = 7087872; g_att = 2362368; g_ff = 4722432
parts_g = [("token table", 38597376), ("position table", 786432), ("attention (12 layers)", 12 * g_att),
           ("feed-forward (12 layers)", 12 * g_ff), ("norms", 12 * 3072 + 1536)]
fig, axs = plt.subplots(2, 1, figsize=(7.2, 7.6))
cols = [AMBER, AMBER, NAVY, BLUE, TEAL]; hatches = ["", "//", "", "..", "xx"]
for ax, parts, tot, name in ((axs[0], parts_t, 2766592, "Tiny, 2,766,592"), (axs[1], parts_g, 124439808, "GPT-2 small, 124,439,808")):
    names = [a for a, _ in parts][::-1]; vals = [b / tot * 100 for _, b in parts][::-1]
    bars = ax.barh(names, vals, color=cols[::-1], edgecolor=SLATE)
    for b, h in zip(bars, hatches[::-1]): b.set_hatch(h)
    for i, v in enumerate(vals): ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=10, color=SLATE)
    ax.set_xlim(0, 75); ax.set_title(name, fontsize=11.5); ax.set_xlabel("share of learned numbers (%)")
fig.tight_layout(); save(fig, os.path.join(OUT, "fig14-09-where-numbers.png"))

# 06 time against depth (one thread)
c = js("cost_timing")
Ls = [int(k) for k in c["depth"]]; ms = [c["depth"][k]["ms"] for k in c["depth"]]
fig, ax = plt.subplots(figsize=(7.2, 5.4))
ax.plot(Ls, ms, "o-", color=PLUM)
for L_, m_ in zip(Ls, ms): ax.annotate(f"{m_:.0f} ms", (L_, m_), (L_ + 0.4, m_ - 60), fontsize=9.5, color=SLATE)
ax.set_xlabel("layers"); ax.set_ylabel("ms for 1,024 tokens (fastest of 5, one thread)"); ax.set_xticks(Ls)
ax.set_title("Each layer adds the same time: about 70 ms here")
save(fig, os.path.join(OUT, "fig14-10-time-depth.png"))

# 07 GPT-2: stream length and relative writes by layer
g = js("gpt2_depth")
A = g["A"]
fig, axs = plt.subplots(2, 1, figsize=(7.2, 7.8), sharex=True)
axs[0].plot(range(12), [a["length"] for a in A], "o-", color=AMBER); axs[0].set_yscale("log")
axs[0].set_ylabel("mean row length entering"); axs[0].set_title("GPT-2 small: the stream grows, the writes stay a fraction of it", fontsize=11.5)
axs[1].plot(range(12), [a["ra"] for a in A], "o-", color=NAVY, label="attention write / row")
axs[1].plot(range(12), [a["rf"] for a in A], "s--", color=BLUE, label="feed-forward write / row")
axs[1].set_ylim(0, 1.6); axs[1].set_xlabel("layer"); axs[1].set_ylabel("write length / row length"); axs[1].legend(fontsize=9.5)
axs[1].annotate(f"layer 0 attention: {A[0]['ra']:.1f}", (0, 1.55), (0.6, 1.4), fontsize=9.5, color=SLATE, arrowprops=dict(arrowstyle="-", color=GREY))
axs[1].set_xticks(range(12))
fig.tight_layout(); save(fig, os.path.join(OUT, "fig14-05-gpt2-writes.png"))

# 08 GPT-2 Logit Lens on 1,023 predictions: agreement with the final top token and gap
B = g["B"]
fig, ax = plt.subplots(figsize=(7.2, 5.6))
ax.plot(range(1, 13), [b["agree"] for b in B[1:]], "o-", color=PLUM, label="top token equals the final top token")
ax2 = ax.twinx(); ax2.plot(range(1, 13), [b["kl"] for b in B[1:]], "s--", color=VERM, label="gap from the final shares (KL)")
ax2.set_ylabel("gap (KL, natural log)", color=VERM); ax2.grid(False); ax2.spines["right"].set_visible(True)
ax.set_xticks(range(1, 13)); ax.set_xticklabels([str(i) for i in range(1, 12)] + ["top"])
ax.set_xlabel("row entering layer"); ax.set_ylabel("share of seats"); ax.set_ylim(0, 1.05)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=9.2, loc="upper left")
ax.set_title("GPT-2's Logit Lens: the answer forms in the last few layers")
save(fig, os.path.join(OUT, "fig14-06-gpt2-lens.png"))

# 09 the running sentence's staircase in GPT-2
C = g["C"]
fig, ax = plt.subplots(figsize=(7.2, 7.4))
ranks = [cc["rank_mat"] for cc in C]
ax.plot(ranks, range(13), "o-", color=AMBER); ax.set_xscale("log"); ax.invert_xaxis()
for i, cc in enumerate(C):
    ax.text(ranks[i] * 0.8, i, f"top: {cc['top'][0]}", fontsize=9, color=NAVY, va="center")
ax.set_yticks(range(13)); ax.set_yticklabels([f"enters {i}" for i in range(12)] + ["top"])
ax.set_xlabel("rank of '␣mat' among 50,257 tokens (1 = first)"); ax.set_xlim(1e5, 1)
ax.set_title("After 'the cat sat on the': '␣mat' climbs, '␣floor' wins")
save(fig, os.path.join(OUT, "fig14-07-gpt2-staircase.png"))

# 10 removing layers from GPT-2: single costs and the extra cost of pairs by distance
single = g["single"]; E = g["E"]
fig, axs = plt.subplots(2, 1, figsize=(7.2, 8.0))
vals = [single[str(l)] for l in range(12)]
bars = axs[0].bar(range(12), vals, color=[VERM if v > 1 else NAVY for v in vals])
axs[0].set_yscale("log"); axs[0].set_ylabel("loss increase"); axs[0].set_xticks(range(12)); axs[0].set_xlabel("layer removed")
axs[0].set_title("One layer removed: layers 0 and 11 matter most", fontsize=11.5)
d = list(range(1, 11)); e = [E[str(k)][0] for k in d]
axs[1].bar(d, e, color=[PLUM if v > 0 else GREY for v in e], edgecolor=SLATE)
axs[1].axhline(0, color=SLATE, lw=0.8); axs[1].set_xticks(d); axs[1].set_xlabel("distance between the two removed layers (layer 0 kept)")
axs[1].set_ylabel("extra cost beyond both singles"); axs[1].set_title("Two removed: neighbours cost more together", fontsize=11.5)
fig.tight_layout(); save(fig, os.path.join(OUT, "fig14-08-gpt2-removal.png"))

# 11 depth against width at one budget: time for one token
sh = c["shape"]
fig, ax = plt.subplots(figsize=(7.2, 5.2))
names = list(sh.keys()); one = [sh[k]["ms_one_token"] for k in names]
bars = ax.bar(names, one, color=[NAVY, BLUE, TEAL], edgecolor=SLATE)
for b, h in zip(bars, ["", "//", ".."]): b.set_hatch(h)
for i, k in enumerate(names): ax.text(i, one[i] + 0.08, f"{one[i]:.2f} ms\n{2 * int(k.split('x')[0])} steps", ha="center", fontsize=9.5, color=SLATE)
ax.set_ylabel("ms for one token (fastest of 5, one thread)"); ax.set_xlabel("layers x width, about 2.37 million numbers in the layers")
ax.set_ylim(0, max(one) * 1.35); ax.set_title("Same budget, different shape: one token's wait")
save(fig, os.path.join(OUT, "fig14-11-depth-width.png"))
print("wrote", sorted(os.listdir(OUT)))
