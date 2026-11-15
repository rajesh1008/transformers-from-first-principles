"""Similarity as agreement: dot products and cosines on the toy table.
Run from 04-code/ch04:  python3 agreement.py"""
import _common
from tiny.tokenizer import TOY_VOCAB
from tiny.embed import TOY_EMBED, dot, length, cosine

E = {w: tuple(int(v) for v in row) for w, row in zip(TOY_VOCAB, TOY_EMBED)}
sentence = ["the", "cat", "sat", "on", "the", "mat"]
for asker in ["sat", "cat"]:
    print(f"{asker} {E[asker]} agrees with each token of the sentence:")
    for w in sentence:
        print(f"  {w:<4} {str(E[w]):<8} dot {dot(E[asker], E[w]):>2}   cosine {cosine(E[asker], E[w]):>6.3f}")
print()
others = [w for w in TOY_VOCAB if w not in ("cat", "<unk>")]
by_dot = sorted(others, key=lambda w: -dot(E["cat"], E[w]))
by_cos = sorted(others, key=lambda w: -cosine(E["cat"], E[w]))
print("cat's nearest neighbours by dot product:", [(w, dot(E['cat'], E[w])) for w in by_dot[:4]])
print("cat's nearest neighbours by cosine:     ", [(w, round(cosine(E['cat'], E[w]), 3)) for w in by_cos[:4]])
print(f"the {E['the']} and a {E['a']}: cosine {cosine(E['the'], E['a']):.3f}, lengths {length(E['the']):.0f} and {length(E['a']):.0f}")
print(f"sat . the = {dot(E['sat'], E['the'])} but sat . a = {dot(E['sat'], E['a'])}: same direction, twice the length, twice the agreement")
print(f"<unk> {E['<unk>']} has length {length(E['<unk>']):.0f}: it agrees with nothing, and its cosine is undefined")
print()
rest = [w for w in TOY_VOCAB if w not in ("sat", "<unk>")]
print("sat against the whole vocabulary (dot, cosine):")
print("  " + ", ".join(f"{w} {dot(E['sat'], E[w])} {cosine(E['sat'], E[w]):.3f}" for w in rest))
