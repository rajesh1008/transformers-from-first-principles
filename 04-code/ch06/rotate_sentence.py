"""The running sentence, every word's arrow turned by its own position (30 degrees per token).
Run from 04-code/ch06:  python3 rotate_sentence.py"""
import _common
from _common import SENTENCE
from tiny.tokenizer import TOY_VOCAB
from tiny.embed import TOY_EMBED, TOY_POS, TOY_TURN, apply_rotary, add_positions, length

words = SENTENCE.split()
X = TOY_EMBED[[TOY_VOCAB.index(w) for w in words]]
R = apply_rotary(X, range(6), speeds=TOY_TURN)
for k, (w, x, r) in enumerate(zip(words, X, R)):
    print(f"{w:>3}@{k}: word row ({x[0]:2.0f}, {x[1]:2.0f}) -> turned {30 * k:3d} deg -> ({r[0]:6.3f}, {r[1]:6.3f})"
          f"   length {length(x):.3f} -> {length(r):.3f}")
print("the two the's:", R[0].round(3).tolist(), "and", R[4].round(3).tolist())
print("different arrows:", len({tuple(r) for r in R.round(6).tolist()}), "of 6")
print("Chapter 5's added positions, for comparison:", add_positions(X, TOY_POS).tolist())
