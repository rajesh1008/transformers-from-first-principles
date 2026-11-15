"""Computed answers to the Chapter 4 exercises (Appendix A).  Run from 04-code/ch04:  python3 solutions.py"""
import _common, _gpt2
import numpy as np
import tiktoken
from tiny.tokenizer import TOY_VOCAB
from tiny.embed import TOY_EMBED, dot, length, cosine
from nudge import nudge, train, nearest

row = {w: TOY_EMBED[i] for i, w in enumerate(TOY_VOCAB)}
f = lambda v: tuple(float(x) for x in np.round(v, 3))
print("Ex1 ids [1, 4, 7] ->", [f(TOY_EMBED[i]) for i in (1, 4, 7)])
print("Ex2 dog + mat =", f(row["dog"] + row["mat"]), "; 2 x on =", f(2 * row["on"]))
print(f"Ex3 length(rabbit) = {length(row['rabbit']):.3f}; length(looked) = {length(row['looked']):.3f}")
print(f"Ex4 rabbit . looked = {dot(row['rabbit'], row['looked']):.0f}; cosine {cosine(row['rabbit'], row['looked']):.3f}")
others = [w for w in TOY_VOCAB if w not in ("hat", "<unk>")]
by_dot = sorted(others, key=lambda w: -dot(row["hat"], row[w]))[:3]
by_cos = sorted(others, key=lambda w: -cosine(row["hat"], row[w]))[:3]
print("Ex5 closest to hat by dot:", [(w, float(round(dot(row['hat'], row[w]), 3))) for w in by_dot],
      "; by cosine:", [(w, float(round(cosine(row['hat'], row[w]), 3))) for w in by_cos])
new, d0, s0 = nudge(row["cat"], row["on"], together=False, step=0.5)
print(f"Ex6 push cat away from on, step 0.5: agreement {d0:.0f}, sure {s0:.3f}, cat -> {f(new)}, "
      f"new agreement {dot(new, row['on']):.3f}")
print("Ex7 dog - cat + the =", f(row["dog"] - row["cat"] + row["the"]),
      "which is the row for", [w for w in TOY_VOCAB if np.array_equal(row[w], row["dog"] - row["cat"] + row["the"])])
n = 50257 * 1024
print(f"Ex8 50,257 x 1,024 = {n:,} numbers = {n * 4 / 1e6:.0f} MB; untied, two tables = {2 * n:,}")
for d in (64, 128, 256):
    print(f"Ex9 width {d}: table {2048 * d:,} numbers = {2048 * d / 1.2e6 * 100:.0f}% of 1.2 million")
for d in (2, 4, 16, 64):
    A, vocab, index = train(d=d, epochs=10, seed=1865)
    nb = nearest(A, vocab, index, "said", 3)
    c = lambda x, y: float(cosine(A[index[x]], A[index[y]]))
    print(f"Ex10 width {d:>2}: cos(said, replied) {c('said', 'replied'):.2f}, cos(said, tea) {c('said', 'tea'):.2f}; "
          f"nearest to said {nb}")
A, vocab, index = train(d=16, epochs=10, seed=1866)
print("Ex11 seed 1866, nearest to queen:", nearest(A, vocab, index, "queen", 5))
print(f"     first 4 numbers for cat: {np.round(A[index['cat']][:4], 2).tolist()}")
enc = tiktoken.get_encoding("gpt2")
W = _gpt2.tensor("wte.weight").astype(float)
N = W / np.linalg.norm(W, axis=1, keepdims=True)
for t in (" Monday", " January"):
    s = N @ N[enc.encode(t)[0]]
    print(f"Ex12 nearest to {t!r}:", [(enc.decode([int(j)]), round(float(s[j]), 2)) for j in np.argsort(-s)[1:8]])
rng = np.random.default_rng(1865)
a, b = rng.normal(size=(5000, 1024)), rng.normal(size=(5000, 1024))
cs = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
print(f"Ex13 random arrows in 1,024 dimensions: average size of cosine {np.abs(cs).mean():.3f}; "
      f"0.8 / sqrt(1024) = {0.8 / 1024 ** 0.5:.3f}")
T = TOY_EMBED.copy(); T[TOY_VOCAB.index("dog")] = T[TOY_VOCAB.index("cat")]
hs = rng.normal(size=(1000, 2))
ties = sum(dot(h, T[1]) == dot(h, T[6]) for h in hs)
print(f"Ex14 dog's row set equal to cat's: cosine {cosine(T[1], T[6]):.0f}; cat and dog score the same for "
      f"{ties} of 1,000 random final vectors")
