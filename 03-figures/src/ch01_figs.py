"""Figures for Chapter 1, computed from the corpus with the same code as the listings.
Run from the project root: python3 03-figures/src/ch01_figs.py"""
import os, sys, numpy as np
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code"))
from style import *
from tiny.corpus import load_text, words
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
setup()
OUT = os.path.join(ROOT, "03-figures", "ch01"); os.makedirs(OUT, exist_ok=True)
P = lambda n: os.path.join(OUT, n)
tokens = words(load_text())
pairs = defaultdict(Counter)
for a, b in zip(tokens, tokens[1:]): pairs[a][b] += 1
def shares(w, k):
    c = pairs[w]; t = sum(c.values()); return [(x, n / t) for x, n in c.most_common(k)]
SENT = "the cat sat on the mat".split()

def strip(ax, y, below=None, color=NAVY, w=13, gap=2.2, x0=4, fs=13, below_color=SLATE, below_fs=10.5):
    for i, t in enumerate(SENT):
        x = x0 + i * (w + gap)
        box(ax, x, y, w, 8, t, color, fs=fs)
        label(ax, x + w / 2, y + 10, f"position {i}", GREY, 8.5)
        if below: label(ax, x + w / 2, y - 3.5, below[i], below_color, below_fs)

# 1.1 finish this sentence (real shares after "at")
def f1():
    s = shares("at", 8)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh([w for w, _ in s][::-1], [v for _, v in s][::-1], color=PLUM)
    for i, (w, v) in enumerate(s[::-1]): ax.text(v + 0.004, i, f"{v:.1%}", va="center", color=SLATE, fontsize=10)
    ax.set_xlabel("share of the times this word follows \u201cat\u201d in the corpus")
    ax.set_title("\u201cThe rabbit looked at \u2026\u201d \u2014 what usually comes next, counted")
    ax.grid(axis="y", visible=False); save(fig, P("fig01-01-finish.png"))

# 1.2 the six-token strip, bare
def f2():
    fig, ax = canvas(10, 2.6); strip(ax, 8)
    label(ax, 2, 24, "The running sentence. It will carry ids, vectors, scores and probabilities in later chapters.", NAVY, 11, ha="left", weight="bold")
    save(fig, P("fig01-02-strip.png"))

# 1.3 the prediction loop
def f3():
    fig, ax = canvas(10, 4.2)
    box(ax, 2, 18, 30, 10, "the cat sat on the", AMBER, sub="text so far", fs=12)
    box(ax, 40, 18, 20, 10, "model", NAVY, sub="scores every word", fs=12)
    box(ax, 68, 18, 14, 10, "shares", PLUM, sub="sum to one", fs=11)
    box(ax, 86, 18, 12, 10, "pick", TEAL, sub="one word", fs=11)
    arrow(ax, 32, 23, 40, 23, SLATE); arrow(ax, 60, 23, 68, 23, SLATE); arrow(ax, 82, 23, 86, 23, SLATE)
    arrow(ax, 92, 18, 17, 17.5, AMBER, rad=-0.35, lw=1.6)
    label(ax, 55, 2.5, "append the picked word and go round again: that loop is all a language model ever does", SLATE, 10)
    label(ax, 2, 40, "Prediction as the whole game", NAVY, 12, ha="left", weight="bold")
    save(fig, P("fig01-03-loop.png"))

# 1.4 checkpoint: top five after "the"
def f4():
    s = shares("the", 5)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar([w for w, _ in s], [v for _, v in s], color=PLUM, width=0.6)
    for i, (w, v) in enumerate(s): ax.text(i, v + 0.0008, f"{pairs['the'][w]} times\n{v:.3f}", ha="center", color=SLATE, fontsize=9.5)
    ax.set_ylabel("share of all words after \u201cthe\u201d"); ax.set_ylim(0, 0.048)
    ax.set_title("Checkpoint: the five most common words after \u201cthe\u201d"); ax.grid(axis="x", visible=False)
    save(fig, P("fig01-04-top5.png"))

# 1.5 pair-count table
def f5():
    ws = ["the", "said", "alice", "queen", "mock", "turtle", "march", "hare", "white", "rabbit"]
    M = np.array([[pairs[a][b] for b in ws] for a in ws])
    fig, ax = plt.subplots(figsize=(8.4, 7)); ax.grid(False)
    im = ax.imshow(np.log1p(M), cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("p", ["#ffffff", "#c9b3dc", PLUM, "#3e2256"]))
    ax.set_xticks(range(len(ws)), ws, rotation=45, ha="right"); ax.set_yticks(range(len(ws)), ws)
    for i in range(len(ws)):
        for j in range(len(ws)):
            if M[i, j]: ax.text(j, i, M[i, j], ha="center", va="center", fontsize=9.5, color="white" if M[i, j] > 20 else NAVY)
    ax.set_xlabel("next word"); ax.set_ylabel("this word")
    ax.set_title("A corner of the pair-count table (blank = never seen)")
    save(fig, P("fig01-05-pair-table.png"))

# 1.6 from prediction to chat
def f6():
    fig, ax = canvas(10, 5)
    label(ax, 2, 48, "From prediction to chat: the same next-word machine, shown different documents", NAVY, 11.5, ha="left", weight="bold")
    box(ax, 2, 26, 44, 15, "", AMBER)
    label(ax, 4, 37.5, "BASE MODEL  \u00b7  continues any document", AMBER, 9.5, ha="left", weight="bold")
    label(ax, 4, 32, "Q: What is the capital of France?\nQ: What is the capital of Spain?\nQ: \u2026", SLATE, 9.5, ha="left", family="JetBrains Mono")
    box(ax, 54, 26, 44, 15, "", NAVY)
    label(ax, 56, 37.5, "INSTRUCTION-TUNED  \u00b7  continues a conversation", NAVY, 9.5, ha="left", weight="bold")
    label(ax, 56, 32, "<user> What is the capital of France?\n<assistant> Paris.\n<end>", SLATE, 9.5, ha="left", family="JetBrains Mono")
    box(ax, 18, 4, 64, 12, "same architecture, same objective: predict the next token", PLUM, sub="instruction tuning only changes which documents it has seen most", fs=11)
    arrow(ax, 24, 26, 40, 16, GREY); arrow(ax, 76, 26, 60, 16, GREY)
    save(fig, P("fig01-06-to-chat.png"))

# 1.7 the pieces, named
def f7():
    fig, ax = canvas(10, 6)
    label(ax, 2, 58, "The pieces, named in plain words (and the chapters that open each one)", NAVY, 11.5, ha="left", weight="bold")
    steps = [("text", "the cat sat on the mat", AMBER, "Ch 2\u20133"), ("tokens", "six ids", AMBER, "Ch 2\u20133"),
             ("embeddings", "a list of numbers per token", NAVY, "Ch 4\u20136"), ("attention", "tokens look at each other", NAVY, "Ch 7\u201311"),
             ("feed-forward + norm", "each token, on its own", NAVY, "Ch 12\u201313"), ("next-token shares", "one share per word", PLUM, "Ch 24, 27")]
    for i, (t, s, c, ch) in enumerate(steps):
        y = 47 - i * 8.2
        box(ax, 22, y, 44, 6.6, t, c, sub=s, fs=10.5)
        label(ax, 70, y + 3.3, ch, GREY, 9.5, ha="left")
        if i: arrow(ax, 44, y + 8.2, 44, y + 6.6, SLATE, ms=10)
    ax.add_patch(FancyBboxPatch((19, 13.0), 50, 17.4, boxstyle="round,pad=0,rounding_size=2", fill=False, ec=NAVY, ls=(0, (4, 3)), lw=1.3))
    label(ax, 17, 21.5, "repeated\n12 times\nin GPT-2", NAVY, 9.5, ha="right")
    save(fig, P("fig01-07-pieces.png"))

# 1.8 the thesis: five small operations
def f8():
    fig, ax = canvas(10, 4.4)
    label(ax, 2, 42, "The thesis: a short list of small operations on lists of numbers, repeated", NAVY, 11.5, ha="left", weight="bold")
    ops = [("lookup", "word \u2192 numbers", AMBER, "Ch 4"), ("dot product", "how much two\nlists agree", NAVY, "Ch 4"),
           ("squash", "scores \u2192 shares", PLUM, "Ch 8"), ("weighted\naverage", "blend by shares", NAVY, "Ch 8"), ("loop", "do it again", TEAL, "Ch 14, 27")]
    for i, (t, s, c, ch) in enumerate(ops):
        x = 2 + i * 19.6
        box(ax, x, 12, 17, 18, t, c, sub=s, fs=11.5)
        label(ax, x + 8.5, 8, ch, GREY, 9.5)
    save(fig, P("fig01-08-thesis.png"))

# 1.9 lab breakage: shares with and without smoothing
def f9():
    cut = int(len(tokens) * 0.9); tr = tokens[:cut]; pr = defaultdict(Counter)
    for a, b in zip(tr, tr[1:]): pr[a][b] += 1
    V = len(set(tokens))
    labels = [f"{a}\u2192{b}" for a, b in zip(SENT, SENT[1:])]
    raw = [pr[a][b] / sum(pr[a].values()) for a, b in zip(SENT, SENT[1:])]
    sm = [(pr[a][b] + 1) / (sum(pr[a].values()) + V) for a, b in zip(SENT, SENT[1:])]
    x = np.arange(len(labels)); fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - 0.2, raw, 0.38, color=CORAL, label="smoothing removed (broken)")
    ax.bar(x + 0.2, sm, 0.38, color=PLUM, label="add-one smoothing")
    for i, v in enumerate(raw):
        if v == 0: ax.text(i - 0.2, 0.004, "0", ha="center", color=CORAL, fontsize=11, weight="bold")
    ax.set_yscale("symlog", linthresh=1e-3); ax.set_xticks(x, labels)
    ax.set_ylabel("share (log scale above 0.001)"); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.set_title("Lab 1: one unseen pair makes the whole sentence impossible", pad=30); ax.grid(axis="x", visible=False)
    save(fig, P("fig01-09-smoothing.png"))

# 1.10 unseen pairs vs amount counted
def f10():
    test = tokens[int(len(tokens) * 0.9):]; tp = list(zip(test, test[1:])); fr, un = [], []
    for f in np.linspace(0.1, 0.9, 9):
        tr = tokens[:int(len(tokens) * f)]; seen = set(zip(tr, tr[1:]))
        fr.append(len(tr)); un.append(sum(p not in seen for p in tp) / len(tp))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(fr, np.array(un) * 100, color=CORAL, marker="o")
    for x_, y_ in zip(fr[::2], un[::2]): ax.text(x_, y_ * 100 + 1.2, f"{y_:.0%}", ha="center", color=SLATE, fontsize=9.5)
    ax.set_xlabel("words counted"); ax.set_ylabel("% of test pairs never seen")
    ax.set_ylim(0, 92); ax.set_title("More text helps slowly: half of real word pairs are still new")
    save(fig, P("fig01-10-unseen.png"))

# 1.11 roadmap
def f11():
    fig, ax = canvas(10, 5.6)
    label(ax, 2, 54, "Where this book goes: seven Parts, one sentence, one model you build", NAVY, 11.5, ha="left", weight="bold")
    parts = [("I", "Words become numbers", "Ch 1\u20136", AMBER), ("II", "Attention, by hand", "Ch 7\u201316", NAVY), ("III", "Learning", "Ch 17\u201326", PLUM),
             ("IV", "Speaking", "Ch 27\u201334", TEAL), ("V", "What the build\nexplains", "Ch 35\u201341", SLATEBLUE), ("VI", "Beyond the vanilla\ntransformer", "Ch 42\u201347", BLUE),
             ("VII", "Capstone and\nwhat comes next", "Ch 48\u201350", GREEN)]
    for i, (n, t, ch, c) in enumerate(parts):
        r, k = divmod(i, 4); x = 2 + k * 24.5; y = 30 - r * 20
        box(ax, x, y, 22.5, 15, "", c)
        label(ax, x + 11.25, y + 11.5, f"PART {n}", c, 9.5, weight="bold"); label(ax, x + 11.25, y + 6.6, t, NAVY, 9.0, weight="bold"); label(ax, x + 11.25, y + 1.8, ch, GREY, 9)
    save(fig, P("fig01-11-roadmap.png"))

if __name__ == "__main__":
    for f in (f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11): f()
    print("ch01 figures:", len(os.listdir(OUT)))
