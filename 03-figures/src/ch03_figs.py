"""Figures for Chapter 3 (Byte-Pair Encoding in Depth), computed with the same code as the listings.
Run from the project root: python3 03-figures/src/ch03_figs.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "04-code"))
from style import *
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from collections import Counter
from tiny.corpus import load_text, load_heldout
from tiny.tokenizer import train_bpe, BPETokenizer, WordTokenizer, TOY_VOCAB, TINY_VOCAB_SIZE, merge_pair
setup()
OUT = os.path.join(ROOT, "03-figures", "ch03"); os.makedirs(OUT, exist_ok=True)
P = lambda n: os.path.join(OUT, n)
SENT = "the cat sat on the mat"
text, sequel = load_text(), load_heldout()
gpt2, o200k = tiktoken.get_encoding("gpt2"), tiktoken.get_encoding("o200k_base")
MERGES = train_bpe(text, TINY_VOCAB_SIZE - 1)
tiny = BPETokenizer(MERGES)
TWENTY = "the cat sat on the mat the rat sat on the hat the cat ate the rat that sat on"
HAND = train_bpe(TWENTY, 256 + 5, split=str.split, min_count=1)

def strip(ax, y, items, color, x0=3, w=None, gap=0.8, h=6.5, fs=10.5, colors=None, subs=None, wscale=2.6, wmin=4.2):
    x = x0
    for i, t in enumerate(items):
        ww = w or max(wmin, wscale * len(t) + 2.0)
        box(ax, x, y, ww, h, t, (colors[i] if colors else color), fs=fs, sub=(subs[i] if subs else None))
        x += ww + gap
    return x

# 3.1 bytes first
def f1():
    fig, ax = canvas(12, 4.6)
    label(ax, 1, 43, "Every text is already a list of bytes: numbers from 0 to 255", NAVY, 12, ha="left", weight="bold")
    chars = ["␣" if c == " " else c for c in SENT]
    bs = list(SENT.encode())
    x = 1
    for c, b in zip(chars, bs):
        box(ax, x, 26, 4.9, 6.5, c, AMBER, fs=11)
        label(ax, x + 2.45, 22.5, str(b), NAVY, 8.6, weight="bold")
        x += 5.35
    label(ax, 1, 35.5, "the running sentence: 22 characters, 22 bytes", AMBER, 10, ha="left", weight="bold")
    label(ax, 1, 15, "a character outside plain English needs more than one byte:", AMBER, 10, ha="left", weight="bold")
    x = 1
    for c in "Où est":
        cc = "␣" if c == " " else c
        bb = list(c.encode())
        ww = 4.9 * len(bb) + 0.45 * (len(bb) - 1)
        box(ax, x, 3, ww, 6.5, cc, CORAL if len(bb) > 1 else AMBER, fs=11)
        for k, b in enumerate(bb):
            label(ax, x + 2.45 + k * 5.35, 0.2, str(b), NAVY, 8.6, weight="bold")
        x += ww + 0.45
    label(ax, x + 3, 6.2, "ù is two bytes, 195 and 185; the curly quote “ is three: 226, 128, 156", SLATE, 9.5, ha="left")
    save(fig, P("fig03-01-bytes.png"))

# 3.2 pair counts on the twenty words
def f2():
    words = TWENTY.split(); c = Counter()
    for w in words:
        for a, b in zip(w, w[1:]): c[a + " " + b] += 1
    items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    names = [k for k, _ in items][::-1]; vals = [v for _, v in items][::-1]
    cols = [CORAL if n == "a t" else NAVY for n in names]
    ax.barh(names, vals, color=cols)
    for i, v in enumerate(vals): ax.text(v + 0.15, i, str(v), va="center", color=SLATE, fontsize=10)
    ax.set_xlabel("how many times the pair occurs inside the twenty words")
    ax.set_title("Counting neighbouring pairs: a t wins with 11")
    ax.grid(axis="y", visible=False)
    save(fig, P("fig03-02-pair-counts.png"))

# 3.3 the first merge
def f3():
    fig, ax = canvas(12, 4.4)
    label(ax, 1, 41, "Merge 1: every a followed by t becomes the single piece at (id 256)", NAVY, 12, ha="left", weight="bold")
    ws = ["the", "cat", "sat", "on", "mat", "rat", "hat", "ate", "that"]
    t0 = BPETokenizer([], specials=()); t1 = BPETokenizer(HAND[:1], specials=())
    label(ax, 1, 31, "before", GREY, 10, ha="left", weight="bold")
    label(ax, 1, 12, "after", GREY, 10, ha="left", weight="bold")
    x = 9
    for w in ws:
        a = [t0.piece(i) for i in t0.encode_chunk(w)]; b = [t1.piece(i) for i in t1.encode_chunk(w)]
        xa = x
        for p in a:
            box(ax, xa, 28, 3.0, 6, p, AMBER, fs=9.5); xa += 3.3
        xb = x
        for p in b:
            ww = 3.0 if len(p) == 1 else 6.3
            box(ax, xb, 9, ww, 6, p, NAVY if p == "at" else AMBER, fs=9.5); xb += ww + 0.3
        arrow(ax, x + 4, 27.5, x + 4, 15.5, GREY, lw=1.0, ms=8)
        x = max(xa, xb) + 1.8
    label(ax, 1, 2, "58 pieces before the merge, 47 after: each merge shortens the text by the number of times its pair occurred", SLATE, 9.5, ha="left")
    save(fig, P("fig03-03-first-merge.png"))

# 3.4 the sentence after 0..5 hand merges
def f4():
    fig, ax = canvas(11, 6.2)
    label(ax, 1, 59, "The running sentence after each of the five hand merges", NAVY, 12, ha="left", weight="bold")
    names = ["no merges", "+ at", "+ th", "+ the", "+ on", "+ sat"]
    for k in range(6):
        t = BPETokenizer(HAND[:k], specials=())
        pieces = []
        for w in SENT.split(): pieces += [t.piece(i) for i in t.encode_chunk(w)]
        y = 49 - k * 9
        label(ax, 1, y + 3.2, names[k], GREY, 9.5, ha="left", weight="bold")
        cols = [NAVY if len(p) > 1 else AMBER for p in pieces]
        end = strip(ax, y, pieces, AMBER, x0=12, colors=cols, h=6.2, fs=10, wscale=2.4, wmin=3.6, gap=0.5)
        label(ax, end + 1.5, y + 3.2, f"{len(pieces)} pieces", SLATE, 9.5, ha="left")
    label(ax, 1, 1, "Spaces are left out here; the real tokenizer keeps them, as the next section shows.", SLATE, 9, ha="left")
    save(fig, P("fig03-04-five-merges.png"))

# 3.5 merge trees
def f5():
    parent = {n: p for p, n, _ in MERGES}
    fig, ax = canvas(12, 6.6)
    label(ax, 1, 64, "Every piece is two earlier pieces glued together, down to single bytes", NAVY, 12, ha="left", weight="bold")
    label(ax, 1, 59, "navy: pieces made by merges, with their ids;  amber: the bytes they are made from", SLATE, 9, ha="left")
    def draw(i, x, y, span):
        p = tiny.piece(i)
        col = AMBER if i < 256 else NAVY
        ww = max(5.0, 2.2 * len(p) + 3)
        box(ax, x - ww / 2, y, ww, 5.4, p, col, fs=10)
        label(ax, x, y - 1.6, str(i), GREY, 7.8)
        if i in parent:
            a, b = parent[i]
            for child, dx in [(a, -span / 2), (b, span / 2)]:
                arrow(ax, x, y - 3, x + dx, y - 7.5, GREY, lw=1.0, ms=7, style="-")
                draw(child, x + dx, y - 13, span / 2)
    draw(tiny.encode(" the")[0], 20, 48, 20)
    draw(tiny.encode(" looked")[0], 76, 48, 40)
    save(fig, P("fig03-05-merge-trees.png"))

# 3.6 the first 24 merges of the novel
def f6():
    fig, ax = plt.subplots(figsize=(9, 6.2))
    items = MERGES[:24][::-1]
    names, vals, cols = [], [], []
    for (a, b), n_id, n in items:
        s = tiny.vocab[n_id]
        try: g = gpt2.encode_single_token(s)
        except KeyError: g = None
        frag = s.decode("utf-8", "replace") in ("\ufffd",) or s in ("“".encode(), "”".encode(), "’".encode())
        lab = tiny.piece(n_id).replace("\ufffd", "(bytes 226,128)")
        names.append(f"{MERGES.index(((a, b), n_id, n)) + 1:>2}. {lab}" + (f"   [GPT-2 {g}]" if g else "   [not in GPT-2]"))
        vals.append(n); cols.append(CORAL if frag else NAVY)
    ax.barh(range(len(vals)), vals, color=cols)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("times the pair occurred in the novel when it was merged")
    ax.set_title("The novel's first 24 merges, in order")
    ax.grid(axis="y", visible=False)
    ax.text(0.98, 0.04, "coral: pieces of curly quotation marks,\nwhich are three bytes each", transform=ax.transAxes, ha="right", fontsize=9, color=CORAL)
    save(fig, P("fig03-06-first-merges.png"))

# 3.7 merge counts fall
def f7():
    full = train_bpe(text, 10**6)
    n = [c for _, _, c in full]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(range(1, len(n) + 1), n, color=NAVY)
    ax.set_yscale("log"); ax.set_xlabel("merge number"); ax.set_ylabel("times the merged pair occurred")
    ax.axvline(1791, color=TEAL, lw=1.4, ls="--"); ax.text(1840, 1200, "Tiny stops here\n(merge 1,791, pair seen 4 times)", color=TEAL, fontsize=9)
    ax.axvline(len(n), color=CORAL, lw=1.4, ls="--"); ax.text(len(n) - 40, 60, f"no pair occurs twice\nafter merge {len(n):,}", color=CORAL, fontsize=9, ha="right")
    ax.set_title("Each merge is rarer than the last")
    save(fig, P("fig03-07-merge-counts.png"))

# 3.8 the vocabulary layout
def f8():
    fig, ax = canvas(12, 3.8)
    label(ax, 1, 35, "Tiny's vocabulary: 2,048 ids in three blocks", NAVY, 12, ha="left", weight="bold")
    box(ax, 2, 14, 24, 10, "ids 0-255", AMBER, sub="the 256 possible bytes", fs=11)
    box(ax, 27, 14, 76, 10, "ids 256-2,046", NAVY, sub="1,791 merged pieces, in the order they were learned", fs=11)
    box(ax, 104, 14, 14, 10, "2,047", PLUM, sub="<|endoftext|>", fs=11)
    label(ax, 3, 10, "bytes:  97 = a    32 = space    226 = first byte of “", SLATE, 9, ha="left")
    cat, last = tiny.encode(" cat")[0], MERGES[-1][1]
    label(ax, 3, 6, f"merges:  256 = ␣t    257 = he    262 = ␣the    {cat} = ␣cat    {last:,} = {tiny.piece(last)}", SLATE, 9, ha="left")
    label(ax, 2, 2, "The merge table and the vocabulary are the same object: knowing the merges tells you every entry.", SLATE, 9.5, ha="left")
    save(fig, P("fig03-08-vocab-layout.png"))

# 3.9 encoding ' cat' step by step
def f9():
    fig, ax = canvas(11, 4.8)
    label(ax, 1, 46, "Encoding ␣cat: apply whichever learned merge is earliest, until none applies", NAVY, 12, ha="left", weight="bold")
    ids = list(" cat".encode()); y = 34; step = 0
    while True:
        pieces = [tiny.piece(i) for i in ids]
        strip(ax, y, pieces, AMBER, x0=30, colors=[NAVY if i > 255 else AMBER for i in ids], h=6, fs=10.5, wscale=2.6, wmin=5)
        label(ax, 52 + 22, y + 3, " ".join(str(i) for i in ids), GREY, 9.5, ha="left")
        if len(ids) < 2: break
        pair = min(zip(ids, ids[1:]), key=lambda p: tiny.ranks.get(p, float("inf")))
        if pair not in tiny.ranks: break
        label(ax, 1, y - 2.5, f"merge {tiny.ranks[pair] - 255}: {tiny.piece(pair[0])} + {tiny.piece(pair[1])}", TEAL, 9.5, ha="left")
        ids = merge_pair(ids, pair, tiny.ranks[pair]); y -= 9.5
    label(ax, 1, 1, "Of all the pairs present, the one learned earliest is merged first; a pair with no merge in the table is left alone.", SLATE, 9, ha="left")
    save(fig, P("fig03-09-encode-steps.png"))

# 3.10 decoding half a character
def f10():
    fig, ax = canvas(11, 4.0)
    label(ax, 1, 37, "Decoding is gluing bytes back together, and a token can hold part of a character", NAVY, 12, ha="left", weight="bold")
    box(ax, 2, 18, 18, 8, "id 258", NAVY, sub="bytes 226, 128", fs=11)
    box(ax, 22, 18, 12, 8, "id 147", AMBER, sub="byte 147", fs=11)
    arrow(ax, 11, 18, 11, 9, CORAL); box(ax, 4, 1, 14, 7, "�", CORAL, sub="decode 258 alone", fs=12)
    arrow(ax, 28, 18, 48, 9, TEAL, rad=0.2); box(ax, 42, 1, 18, 7, "–", TEAL, sub="decode 258, 147", fs=12)
    label(ax, 64, 22, "The en dash never appears in the novel, so it has no token\nof its own. Its three bytes become id 258, a piece shared\nwith the curly quotes, plus the single byte 147.", SLATE, 9.5, ha="left")
    save(fig, P("fig03-10-broken-bytes.png"))

# 3.11 the vocabulary dial
DIAL = [(256, 2), (300, 2), (500, 2), (1000, 2), (2048, 2), (3000, 2), (5000, 1), (50000, 1)]
def dial_rows():
    rows = []
    for size, mc in DIAL:
        t = BPETokenizer(train_bpe(text, size - 1, min_count=mc))
        rows.append((len(t), len(t.encode(text)), len(t.encode(sequel)), t))
    return rows
ROWS = dial_rows()
def f11():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    v = [r[0] for r in ROWS]
    ax.plot(v, [r[1] for r in ROWS], "-o", color=NAVY, label="the novel (training text)")
    ax.plot(v, [r[2] for r in ROWS], "-o", color=AMBER, label="the sequel (never seen)")
    ax.scatter([gpt2.n_vocab], [len(gpt2.encode(text))], color=NAVY, marker="D", s=50, zorder=5)
    ax.scatter([gpt2.n_vocab], [len(gpt2.encode(sequel))], color=AMBER, marker="D", s=50, zorder=5)
    ax.text(gpt2.n_vocab, 62000, "GPT-2\n50,257", ha="center", fontsize=9, color=SLATE)
    top = ROWS[-1][0]
    ax.axvline(top, color=CORAL, ls="--", lw=1.3); ax.text(top * 1.07, 140000, f"the novel runs out\nof pairs at {top:,}", color=CORAL, fontsize=9)
    ax.axvline(2048, color=TEAL, ls="--", lw=1.3); ax.text(2000, 140000, "Tiny\n2,048", color=TEAL, fontsize=9, ha="right")
    ax.set_xscale("log"); ax.set_xlabel("vocabulary size (each grid line is ten times the last)")
    ax.set_ylabel("tokens"); ax.set_title("Turning the dial: more entries, fewer tokens, with diminishing returns")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    save(fig, P("fig03-11-dial.png"))

# 3.12 the sentence at several sizes
def f12():
    fig, ax = canvas(13, 5.6)
    label(ax, 1, 53, "The running sentence as the dial turns", NAVY, 12, ha="left", weight="bold")
    for k, r in enumerate([ROWS[0], ROWS[1], ROWS[2], ROWS[3], ROWS[4]]):
        t = r[3]; ids = t.encode(SENT); y = 43 - k * 10
        label(ax, 1, y + 3, f"{r[0]:,} entries", GREY, 9.5, ha="left", weight="bold")
        strip(ax, y, [t.piece(i) for i in ids], AMBER, x0=15, colors=[NAVY if i > 255 else AMBER for i in ids],
              h=6, fs=9.5, wscale=2.2, wmin=3.4, gap=0.4)
        label(ax, 125, y + 3, f"{len(ids)}", SLATE, 10, weight="bold")
    label(ax, 125, 50, "tokens", GREY, 9)
    label(ax, 1, 1, "From 1,000 entries on, the sentence stops changing: ␣mat never occurs in the novel, so it stays two pieces.", SLATE, 9, ha="left")
    save(fig, P("fig03-12-sentence-dial.png"))

# 3.13 Tiny against GPT-2 and o200k
def f13():
    names = ["Tiny\n2,048", "GPT-2\n50,257", "o200k\n200,019"]
    a = [len(tiny.encode(text)), len(gpt2.encode(text)), len(o200k.encode(text))]
    b = [len(tiny.encode(sequel)), len(gpt2.encode(sequel)), len(o200k.encode(sequel))]
    x = np.arange(3); fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - 0.2, a, 0.4, color=NAVY, label="the novel"); ax.bar(x + 0.2, b, 0.4, color=AMBER, label="the sequel")
    for i in range(3):
        ax.text(x[i] - 0.2, a[i] + 600, f"{a[i]:,}", ha="center", fontsize=9); ax.text(x[i] + 0.2, b[i] + 600, f"{b[i]:,}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("tokens"); ax.legend(loc="upper right")
    ax.set_title("A vocabulary 25 times smaller, trained on one book")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax.grid(axis="x", visible=False)
    save(fig, P("fig03-13-tiny-vs-gpt2.png"))

# 3.14 bytes per token
def f14():
    lt = Counter(len(tiny.vocab[i]) for i in tiny.encode(text))
    lg = Counter(len(gpt2.decode_single_token_bytes(i)) for i in gpt2.encode(text))
    L = range(1, 13); nt, ng = sum(lt.values()), sum(lg.values())
    fig, ax = plt.subplots(figsize=(9, 4.2)); x = np.array(list(L))
    ax.bar(x - 0.2, [lt[k] / nt for k in L], 0.4, color=NAVY, label="Tiny")
    ax.bar(x + 0.2, [lg[k] / ng for k in L], 0.4, color=GREY, label="GPT-2")
    ax.set_xticks(x); ax.set_xlabel("bytes in the token (a leading space counts as one)")
    ax.set_ylabel("share of the novel's tokens"); ax.legend()
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("How long are the tokens? Encoding the novel")
    ax.grid(axis="x", visible=False)
    save(fig, P("fig03-14-token-lengths.png"))

# 3.15 residues: made often, used rarely
def f15():
    used = Counter(tiny.encode(text)); made = {n_id: n for _, n_id, n in MERGES}
    xs = np.array([made[i] for i in made]); ys = np.array([used[i] for i in made])
    fig, ax = plt.subplots(figsize=(8, 5))
    m = ys > 0
    ax.scatter(xs[m], ys[m], s=9, color=NAVY, alpha=0.55, label="used in the final encoding")
    ax.scatter(xs[~m], np.full((~m).sum(), 0.6), s=16, color=CORAL, label=f"never used ({(~m).sum()} pieces)")
    lim = [1, 5000]; ax.plot(lim, lim, color=GREY, lw=1, ls="--")
    ax.text(1500, 2600, "used as often\nas when made", fontsize=8.5, color=GREY, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(2, 5000); ax.set_ylim(0.4, 5000)
    ax.set_xlabel("times the pair occurred when the piece was made"); ax.set_ylabel("times the piece is used\nwhen the novel is encoded")
    for s in ["lice", "ght", "ryph"]:
        i = next(k for k in made if tiny.vocab[k].decode("utf-8", "replace") == s)
        ax.annotate(s, (made[i], 0.6), xytext=(made[i] * 1.1, 2.0), fontsize=9, color=CORAL, arrowprops=dict(arrowstyle="-", color=CORAL, lw=0.8))
    ax.set_title("Scaffolding: pieces that were built only to be built on")
    ax.legend(loc="upper left")
    save(fig, P("fig03-15-residues.png"))

# 3.16 the running sentence under two vocabularies
def f16():
    fig, ax = canvas(12, 4.6)
    label(ax, 1, 44, "The running sentence under the toy vocabulary and under Tiny's real one", NAVY, 12, ha="left", weight="bold")
    toy = WordTokenizer(TOY_VOCAB)
    label(ax, 1, 36, "toy (12 words, chosen by hand): 6 tokens", AMBER, 10, ha="left", weight="bold")
    for i, (w, d) in enumerate(zip(SENT.split(), toy.encode(SENT))):
        box(ax, 2 + i * 16, 26, 14, 7, w, AMBER, fs=11); label(ax, 9 + i * 16, 23.3, str(d), NAVY, 9.5, weight="bold")
    label(ax, 1, 16, "Tiny BPE (2,048 entries, learned from the novel): 7 tokens", NAVY, 10, ha="left", weight="bold")
    ids = tiny.encode(SENT)
    for i, d in enumerate(ids):
        box(ax, 2 + i * 16, 5, 14, 7, tiny.piece(d), NAVY, fs=11); label(ax, 9 + i * 16, 2.3, str(d), NAVY, 9.5, weight="bold")
    save(fig, P("fig03-16-two-vocabularies.png"))

# 3.17 how many tokens a number needs
def tiny_saved():
    from tiny.tokenizer import load_bpe
    return load_bpe(os.path.join(ROOT, "04-code", "tiny", "tiny_bpe_2048.txt"))

def f17():
    t2 = tiny_saved()
    encs = [("Tiny", lambda s: len(t2.encode(s)), NAVY), ("GPT-2", lambda s: len(gpt2.encode(s)), GREY),
            ("o200k", lambda s: len(o200k.encode(s)), AMBER)]
    K = range(1, 6); x = np.array(list(K)); fig, ax = plt.subplots(figsize=(9, 4.2))
    for j, (name, f, col) in enumerate(encs):
        c = Counter(f(f" {n}") for n in range(10000))
        vals = [c[k] for k in K]
        ax.bar(x + (j - 1) * 0.27, vals, 0.27, color=col, label=name)
        for k, v in zip(K, vals):
            if v: ax.text(k + (j - 1) * 0.27, v + 120, f"{v:,}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xlabel("tokens needed for the number, its leading space included")
    ax.set_ylabel("how many of the numbers 0 to 9,999"); ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:,.0f}"))
    ax.set_title("Ten thousand numbers through three vocabularies"); ax.grid(axis="x", visible=False)
    save(fig, P("fig03-17-number-tokens.png"))

# 3.18 where GPT-2 cuts every four-digit number
def f18():
    pat = lambda n: "+".join(str(sum(ch.isdigit() for ch in gpt2.decode([i]))) for i in gpt2.encode(f" {n}"))
    order = ["2+2", "1+3", "3+1", "1+2+1", "4"]
    cols = [NAVY, AMBER, TEAL, CORAL, PLUM]
    grid = np.zeros((90, 100))
    for n in range(1000, 10000):
        grid[(n - 1000) // 100, n % 100] = order.index(pat(n))
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(9, 6.4))
    ax.imshow(grid, cmap=ListedColormap(cols), aspect="auto", interpolation="nearest", vmin=-0.5, vmax=4.5,
              extent=[-0.5, 99.5, 99.5, 9.5])
    ax.set_xlabel("last two digits"); ax.set_ylabel("first two digits")
    ax.set_xticks([0, 20, 40, 60, 80, 99]); ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 99])
    ax.grid(False)
    for c, o in zip(cols, order):
        ax.bar(0, 0, color=c, label=f"{o} digits")
    ax.set_xlim(-0.5, 99.5); ax.set_ylim(99.5, 9.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False)
    ax.set_title("How GPT-2 cuts each number from 1000 to 9999 (written after a space)")
    save(fig, P("fig03-18-four-digit-cuts.png"))

# 3.19 one line of code through three vocabularies
def f19():
    t2 = tiny_saved()
    line = "            counts[(a, b)] = counts.get((a, b), 0) + 1\n"
    rows = [("GPT-2", [gpt2.decode([i]) for i in gpt2.encode(line)], GREY),
            ("Tiny", [t2.vocab[i].decode("utf-8", "replace") for i in t2.encode(line)], NAVY),
            ("o200k", [o200k.decode([i]) for i in o200k.encode(line)], AMBER)]
    fig, ax = canvas(14, 4.3)
    label(ax, 1, 40.5, "One line of Python, indented by twelve spaces", NAVY, 12, ha="left", weight="bold")
    y = 27.5
    for name, ps, col in rows:
        shown = [p.replace(" ", "\u2423").replace("\n", "\u21b5") for p in ps]
        label(ax, 1, y + 6.4 + (0 if name else 0), f"{name}: {len(ps)} tokens", col, 10, ha="left", weight="bold")
        x = 1; yy = y
        for t in shown:
            ww = max(2.6, 1.55 * len(t) + 1.1)
            if x + ww > 139: x = 1; yy -= 6
            box(ax, x, yy, ww, 5, t, col, fs=8.5, lw=1.2, r=0.8)
            x += ww + 0.35
        y = yy - 12.5
    save(fig, P("fig03-19-code-line.png"))

import matplotlib
for f in [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19]:
    f()
print("ch03 figures:", len(os.listdir(OUT)))
