"""Section 5.2: add a position row to every token row.  Run: python3 add_position.py"""
import _common
from _common import SENTENCE, SCRAMBLED
from tiny.tokenizer import TOY_VOCAB
from tiny.embed import TOY_EMBED, TOY_POS, add_rows, add_positions

def show(text):
    words = text.split()
    X = TOY_EMBED[[TOY_VOCAB.index(w) for w in words]]
    Y = add_positions(X, TOY_POS)
    print(f"{text!r}")
    for k, (w, x, p, y) in enumerate(zip(words, X, TOY_POS, Y)):
        print(f"  position {k}  {w:<4} ({x[0]:>4.1f}, {x[1]:>4.1f}) + ({p[0]:>4.1f}, {p[1]:>4.1f}) = ({y[0]:>4.1f}, {y[1]:>4.1f})")
    return Y

Y = show(SENTENCE)
Z = show(SCRAMBLED)
print("different arrows in the first sentence:", len({tuple(r) for r in Y.tolist()}), "of 6 (was 5)")
print("the two 'the's:", Y[0].tolist(), "and", Y[4].tolist(), "-> same?", Y[0].tolist() == Y[4].tolist())
print("same bag as the scrambled sentence:", sorted(Y.tolist()) == sorted(Z.tolist()))
print("average row:", [round(float(v), 3) for v in Y.mean(axis=0)], "vs", [round(float(v), 3) for v in Z.mean(axis=0)])

# Attaching instead of adding: rows twice as long
print("attaching instead of adding gives 4 numbers per row, e.g. 'the' at 0:", TOY_EMBED[0].tolist() + TOY_POS[0].tolist())

# Collisions: can a different word at a different place land on the same arrow?
seen = {}
for w_id, w in enumerate(TOY_VOCAB[:11]):
    for k in range(6):
        seen.setdefault(tuple(add_rows(TOY_EMBED[w_id], TOY_POS[k])), []).append(f"{w}@{k}")
clashes = [v for v in seen.values() if len(v) > 1]
print(f"word-and-place pairs: {11 * 6}; arrows shared by two or more pairs: {len(clashes)}; first three: {clashes[:3]}")
