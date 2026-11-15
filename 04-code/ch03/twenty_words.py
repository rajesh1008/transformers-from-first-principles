"""The first merges by hand, on twenty words.  Spaces are ignored here: each word is
merged on its own, so we can count with pencil and paper.
Run from 04-code/ch03:  python3 twenty_words.py"""
import _common
from collections import Counter
from tiny.tokenizer import train_bpe, BPETokenizer

TEXT = ("the cat sat on the mat the rat sat on the hat "
        "the cat ate the rat that sat on")
words = TEXT.split()
print(f"{len(words)} words; distinct words and counts: {dict(Counter(words))}")

pieces = [list(w) for w in words]
pairs = Counter()
for w in pieces:
    for a, b in zip(w, w[1:]):
        pairs[a + b] += 1
print("\npair counts before any merge:")
for p, n in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"  {p[0]} {p[1]}  {n:>2}")

merges = train_bpe(TEXT, 256 + 5, split=str.split, min_count=1)
tok = BPETokenizer(merges, specials=())
print("\nfive merges:")
for k, ((a, b), new_id, n) in enumerate(merges, 1):
    print(f"  merge {k}: {tok.piece(a)!r} + {tok.piece(b)!r} -> {tok.piece(new_id)!r}  (seen {n} times, new id {new_id})")

for k in range(6):
    t = BPETokenizer(merges[:k], specials=())
    segs = [[t.piece(i) for i in t.encode_chunk(w)] for w in ["the", "cat", "sat", "on", "mat", "that"]]
    total = sum(len(t.encode_chunk(w)) for w in words)
    print(f"after {k} merges: " + "  ".join("|".join(s) for s in segs) + f"   (the twenty words: {total} pieces)")
