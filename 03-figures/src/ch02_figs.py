"""Figures for Chapter 2 (Tokens), computed with the same code as the listings.
Run from the project root: python3 03-figures/src/ch02_figs.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code"))
from style import *
import matplotlib.pyplot as plt
import tiktoken
from collections import Counter
from tiny.corpus import load_text
from tiny.tokenizer import WordTokenizer, CharTokenizer, TOY_VOCAB, split_words
setup()
OUT = os.path.join(ROOT, "03-figures", "ch02"); os.makedirs(OUT, exist_ok=True)
P = lambda n: os.path.join(OUT, n)
SENT = "the cat sat on the mat"; W = SENT.split()
gpt2, o200k = tiktoken.get_encoding("gpt2"), tiktoken.get_encoding("o200k_base")
toy = WordTokenizer(TOY_VOCAB); text = load_text()

def row(ax, y, items, color, w, gap=1.0, x0=4, fs=11, h=7, sub=None):
    x = x0
    for i, t in enumerate(items):
        ww = w(t) if callable(w) else w
        box(ax, x, y, ww, h, t, color, fs=fs, sub=(sub[i] if sub else None))
        x += ww + gap
    return x

# 2.1 three cuts
def f1():
    fig, ax = canvas(11, 5.4)
    label(ax, 2, 51, "Three ways to cut the same sentence", NAVY, 12, ha="left", weight="bold")
    label(ax, 2, 44, "characters: 22 pieces", AMBER, 10.5, ha="left", weight="bold")
    row(ax, 34, ["␣" if c == " " else c for c in SENT], AMBER, 4.2, gap=0.55, fs=10, h=7)
    label(ax, 2, 28, "words: 6 pieces", AMBER, 10.5, ha="left", weight="bold")
    row(ax, 18, W, AMBER, 15, gap=2, fs=12)
    label(ax, 2, 12, "GPT-2 pieces: 6 pieces, each carrying its leading space", NAVY, 10.5, ha="left", weight="bold")
    pieces = [gpt2.decode([t]).replace(" ", "␣") for t in gpt2.encode(SENT)]
    row(ax, 2, pieces, NAVY, 15, gap=2, fs=12)
    save(fig, P("fig02-01-three-cuts.png"))

# 2.2 Signature 1: the six-token strip with toy ids
def f2():
    fig, ax = canvas(10, 3.0)
    ids = toy.encode(SENT)
    for i, (t, d) in enumerate(zip(W, ids)):
        x = 4 + i * 15.2
        box(ax, x, 12, 13, 8, t, NAVY, fs=13)
        label(ax, x + 6.5, 22.5, f"position {i}", GREY, 8.5)
        box(ax, x + 3, 2, 7, 6, str(d), AMBER, fs=12)
        arrow(ax, x + 6.5, 12, x + 6.5, 8.2, AMBER, lw=1.4, ms=10)
    label(ax, 2, 28, "Signature 1, the six-token strip: the running sentence with its toy-vocabulary ids", NAVY, 11, ha="left", weight="bold")
    save(fig, P("fig02-02-strip-ids.png"))

# 2.3 the toy vocabulary as a lookup table
def f3():
    fig, ax = canvas(8, 5.4)
    label(ax, 2, 51, "The toy vocabulary: twelve entries, position is the id", NAVY, 11.5, ha="left", weight="bold")
    for i, w in enumerate(TOY_VOCAB):
        r, c = i % 6, i // 6; x = 6 + c * 36; y = 40 - r * 7
        col = CORAL if w == "<unk>" else (NAVY if w in W else AMBER)
        box(ax, x, y, 8, 5.6, str(i), GREY, fs=10); box(ax, x + 9, y, 22, 5.6, w, col, fs=11)
    label(ax, 6, 3, "navy: words in the running sentence   amber: other words   coral: the catch-all for everything else", SLATE, 9, ha="left")
    save(fig, P("fig02-03-toy-vocab.png"))

# 2.4 encode / decode round trip
def f4():
    fig, ax = canvas(10, 4.2)
    box(ax, 2, 18, 26, 10, "the cat sat on the mat", AMBER, sub="text", fs=11)
    box(ax, 37, 18, 18, 10, "encode", TEAL, sub="look up each word", fs=11)
    box(ax, 64, 18, 30, 10, "[0, 1, 2, 3, 0, 4]", AMBER, sub="ids: what the model sees", fs=11)
    arrow(ax, 28, 23, 37, 23); arrow(ax, 55, 23, 64, 23)
    arrow(ax, 79, 18, 15, 17.5, TEAL, rad=-0.25, lw=1.5)
    label(ax, 47, 2.5, "decode: look each id up the other way and join the words", TEAL, 10)
    label(ax, 2, 39, "A tokenizer is two lookups that undo each other", NAVY, 11.5, ha="left", weight="bold")
    save(fig, P("fig02-04-roundtrip.png"))

# 2.5 corpus coverage by the toy vocabulary
def f5():
    ids = toy.encode(text); c = Counter(ids)
    names = [toy.itos[i] for i, _ in c.most_common()]; vals = [n for _, n in c.most_common()]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    cols = [CORAL if n == "<unk>" else (NAVY if n in W else AMBER) for n in names]
    ax.bar(names, vals, color=cols); ax.set_yscale("log")
    for i, v in enumerate(vals): ax.text(i, v * 1.12, f"{v:,}", ha="center", fontsize=9, color=SLATE)
    ax.set_ylabel("pieces in the novel (log scale)")
    ax.set_title("The novel through the toy vocabulary: 92% of it is <unk>")
    ax.grid(axis="x", visible=False); save(fig, P("fig02-05-coverage.png"))

# 2.6 sequence length of the novel three ways
def f6():
    n = [len(text), len(split_words(text)), len(gpt2.encode(text, disallowed_special=()))]
    v = [len(set(text)), len(set(split_words(text))), gpt2.n_vocab]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    labs = ["characters", "words", "GPT-2 tokens"]
    axs[0].bar(labs, n, color=[AMBER, AMBER, NAVY]); axs[0].set_title("How long the novel is")
    for i, x in enumerate(n): axs[0].text(i, x * 1.02, f"{x:,}", ha="center", fontsize=10, color=SLATE)
    axs[1].bar(labs, v, color=[AMBER, AMBER, NAVY]); axs[1].set_yscale("log"); axs[1].set_title("How many different pieces")
    for i, x in enumerate(v): axs[1].text(i, x * 1.15, f"{x:,}", ha="center", fontsize=10, color=SLATE)
    for a in axs: a.grid(axis="x", visible=False)
    save(fig, P("fig02-06-lengths.png"))

# 2.7 the trade-off: vocabulary size vs pieces per sentence
def f7():
    pts = [("characters", len(set(text)), len(SENT)), ("toy words", 12, 6), ("novel's words", len(set(split_words(text))), 6),
           ("GPT-2", 50257, 6), ("o200k", 200019, len(o200k.encode(SENT)))]
    novel = [("characters", len(set(text)), len(text)), ("novel's words", len(set(split_words(text))), len(split_words(text))),
             ("GPT-2", 50257, len(gpt2.encode(text, disallowed_special=()))), ("o200k", 200019, len(o200k.encode(text, disallowed_special=())))]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = [p[1] for p in novel]; ys = [p[2] for p in novel]
    ax.plot(xs, ys, "o-", color=PLUM)
    for name, x, y in novel: ax.annotate(f"{name}\n{y:,} pieces", (x, y), textcoords="offset points", xytext=(10, -4) if name == "characters" else (-30, 12), fontsize=9.5, color=SLATE)
    ax.set_xscale("log"); ax.set_xlim(40, 600000); ax.set_ylim(25000, 160000); ax.set_xlabel("vocabulary size (log scale)"); ax.set_ylabel("pieces needed for the whole novel")
    ax.set_title("Vocabulary size against length: the whole novel, four ways"); save(fig, P("fig02-07-tradeoff.png"))

# 2.8 strawberry under three tokenizers
def f8():
    fig, ax = canvas(10, 6.4)
    label(ax, 2, 61, "What the model is handed when you type \u201cstrawberry\u201d", NAVY, 11.5, ha="left", weight="bold")
    y = 50
    for name in ["gpt2", "cl100k_base", "o200k_base"]:
        enc = tiktoken.get_encoding(name)
        for t in ["strawberry", " strawberry"]:
            ids = enc.encode(t); ps = [enc.decode([i]).replace(" ", "␣") for i in ids]
            label(ax, 2, y + 3, f"{name}, {t!r}", GREY, 9, ha="left")
            x = 30
            for p, i in zip(ps, ids):
                ww = 5 + 2.2 * len(p); box(ax, x, y, ww, 6, p, NAVY, fs=10, sub=None); label(ax, x + ww / 2, y - 2, str(i), AMBER, 8.5); x += ww + 1.5
            y -= 9.6
    save(fig, P("fig02-08-strawberry.png"))

# 2.9 counting letters with a blindfold
def f9():
    fig, ax = canvas(10, 3.6)
    label(ax, 2, 33, "Counting letters with a blindfold: the model receives ids, not letters", NAVY, 11.5, ha="left", weight="bold")
    letters = list("strawberry")
    for i, c in enumerate(letters):
        box(ax, 4 + i * 5.4, 19, 4.6, 6, c, CORAL if c == "r" else AMBER, fs=11)
    label(ax, 61, 22, "what you see: ten letters, three r", SLATE, 10, ha="left")
    for j, (p, i) in enumerate(zip(["st", "raw", "berry"], [301, 1831, 8396])):
        box(ax, 4 + j * 18, 4, 15, 7, str(i), NAVY, fs=12)
    label(ax, 61, 7.5, "what the model sees: three ids", SLATE, 10, ha="left")
    save(fig, P("fig02-09-blindfold.png"))

# 2.10 the sentence in five languages
def f10():
    langs = ["English", "German", "Greek", "Hindi", "Japanese"]
    S = ["the cat sat on the mat", "die Katze saß auf der Matte", "η γάτα κάθισε στο χαλάκι", "बिल्ली चटाई पर बैठी", "猫はマットの上に座った"]
    g = [len(gpt2.encode(s)) for s in S]; o = [len(o200k.encode(s)) for s in S]
    fig, ax = plt.subplots(figsize=(9, 4.4)); import numpy as np
    x = np.arange(len(langs)); ax.bar(x - 0.2, g, 0.4, color=NAVY, label="GPT-2 (2019, 50,257 tokens)"); ax.bar(x + 0.2, o, 0.4, color=TEAL, label="o200k (2024, 200,019 tokens)")
    for i in range(len(langs)): ax.text(x[i] - 0.2, g[i] + 0.5, str(g[i]), ha="center", fontsize=9.5); ax.text(x[i] + 0.2, o[i] + 0.5, str(o[i]), ha="center", fontsize=9.5)
    ax.set_xticks(x, langs); ax.set_ylabel("tokens for the same sentence"); ax.legend()
    ax.set_title("The same sentence costs different amounts in different languages"); ax.grid(axis="x", visible=False)
    save(fig, P("fig02-10-languages.png"))

# 2.11 bytes per character by script
def f11():
    chars = [("a", "Latin a"), ("ß", "German ß"), ("γ", "Greek gamma"), ("ब", "Devanagari ba"), ("猫", "Japanese neko")]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    b = [len(c.encode("utf-8")) for c, _ in chars]
    ax.barh([n for _, n in chars][::-1], b[::-1], color=AMBER)
    for i, v in enumerate(b[::-1]): ax.text(v + 0.05, i, f"{v} byte{'s' if v > 1 else ''}", va="center", fontsize=10)
    ax.set_xlim(0, 3.8); ax.set_xlabel("bytes needed in UTF-8"); ax.set_title("Before any tokenizer runs, some scripts already cost more")
    ax.grid(axis="y", visible=False); save(fig, P("fig02-11-bytes.png"))

# 2.12 special tokens in a document
def f12():
    fig, ax = canvas(10, 3.4)
    label(ax, 2, 31, "Special tokens mark structure; they are not text anyone typed", NAVY, 11.5, ha="left", weight="bold")
    items = [("the cat sat on the mat", AMBER, 30), ("<|endoftext|>", PLUM, 18), ("The rabbit looked at", AMBER, 28)]
    x = 3
    for t, c, w in items:
        box(ax, x, 12, w, 9, t, c, fs=10.5, sub=("id 50256" if c == PLUM else "ordinary pieces")); x += w + 3
    label(ax, 50, 4, "typed as text, the same characters become seven ordinary pieces: < | end of text | >", CORAL, 9.5)
    save(fig, P("fig02-12-special.png"))

# 2.13 what a tokenizer cannot know
def f13():
    fig, ax = canvas(10, 4.0)
    label(ax, 2, 37, "To the model, these are five unrelated numbers", NAVY, 11.5, ha="left", weight="bold")
    for i, t in enumerate(["cat", " cat", "Cat", " Cat", " CAT"]):
        x = 3 + i * 19
        box(ax, x, 18, 16, 8, repr(t), AMBER, fs=11); box(ax, x + 3, 4, 10, 7, str(gpt2.encode(t)[0]), NAVY, fs=11)
        arrow(ax, x + 8, 18, x + 8, 11.2, SLATE, lw=1.3, ms=10)
    label(ax, 50, 30, "the ids carry no spelling, no meaning and no hint that they are the same word", SLATE, 9.5)
    save(fig, P("fig02-13-cannot-know.png"))

if __name__ == "__main__":
    for f in (f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13): f()
    print("ch02 figures:", len(os.listdir(OUT)))
