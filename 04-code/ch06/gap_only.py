"""Why the agreement depends only on the gap: the same word pair, the same gap, different places.
Run from 04-code/ch06:  python3 gap_only.py"""
import _common, math
import numpy as np
from _common import SENTENCE
from tiny.tokenizer import TOY_VOCAB
from tiny.embed import TOY_EMBED, TOY_TURN, apply_rotary, add_positions, agreement_table, dot, rotate_pair, clock

ROW = {w: TOY_EMBED[TOY_VOCAB.index(w)] for w in TOY_VOCAB}
def turned(word, pos):
    return apply_rotary([ROW[word]], [pos], speeds=TOY_TURN)[0]

for m in [0, 4, 9, 100]:
    a, b = turned("the", m), turned("cat", m + 1)
    print(f"the@{m:<3} . cat@{m + 1:<3} = {dot(a, b):6.3f}")
for m in [0, 4, 9, 100]:
    a, b = turned("sat", m), turned("mat", m + 3)
    print(f"sat@{m:<3} . mat@{m + 3:<3} = {dot(a, b):6.3f}")
# the whole rule: turn both by their own positions = turn only the second one, by the gap
a = ROW["the"]; c = ROW["cat"]
x, y = rotate_pair(c[0], c[1], math.radians(30 * 1))
print(f"the (not turned) . cat turned by the gap of 1 = {dot(a, (x, y)):6.3f}")

words = SENTENCE.split()
X = TOY_EMBED[[TOY_VOCAB.index(w) for w in words]]
def table(start, how):
    if how == "rotary":
        rows = apply_rotary(X, range(start, start + 6), speeds=TOY_TURN)
    else:   # Chapter 5's way: add the toy clock hand for each position to the word row
        rows = add_positions(X, [clock(k) for k in range(start, start + 6)])
    return agreement_table(rows)
for how in ["rotary", "added"]:
    t0, t7 = table(0, how), table(7, how)
    changed = int((np.abs(t0 - t7) > 1e-9).sum())
    print(f"{how:>6}: sentence at positions 0-5 vs 7-12: {changed} of 36 agreements change")
# why adding breaks the gap rule: (word + place) . (word + place) is four dot products, and only one is gap-only
for m in [0, 7]:
    w1, w2, p1, p2 = ROW["the"], ROW["cat"], np.array(clock(m)), np.array(clock(m + 1))
    parts = [dot(w1, w2), dot(w1, p2), dot(p1, w2), dot(p1, p2)]
    print(f"added: the@{m} . cat@{m + 1} = {sum(parts):.3f} = word.word {parts[0]:.3f} + word.place {parts[1]:.3f}"
          f" + place.word {parts[2]:.3f} + place.place {parts[3]:.3f}")
