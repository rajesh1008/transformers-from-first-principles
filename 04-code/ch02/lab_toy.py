"""Lab 2: tokenize the novel with the toy vocabulary, then break it with an unknown word.
Run from 04-code/ch02:  python3 lab_toy.py"""
import _common, traceback
from collections import Counter
from _common import SENTENCE
from tiny.corpus import load_text
from tiny.tokenizer import WordTokenizer, TOY_VOCAB

tok = WordTokenizer(TOY_VOCAB)
text = load_text()
ids = tok.encode(text)
unk = tok.stoi["<unk>"]
known = sum(1 for i in ids if i != unk)
print(f"pieces in the novel: {len(ids):,}; known to the toy vocabulary: {known:,} ({known/len(ids):.1%})")
print("how often each toy word occurs:")
for i, n in Counter(ids).most_common():
    print(f"  {tok.itos[i]:<7} {n:>6,}")
start = text.index("Alice was beginning")
first = " ".join(text[start:start + 120].split()[:15])
print(f"\nthe story's first fifteen words: {first!r}")
print(f"as toy ids:             {tok.encode(first)}")
print(f"decoded:                {tok.decode(tok.encode(first))!r}")

print("\nBREAKAGE: strict mode, no <unk> fallback, on 'the cats sat on the mat'")
try:
    tok.encode("the cats sat on the mat", strict=True)
except KeyError:
    print("  " + traceback.format_exc().strip().splitlines()[-1])
print(f"\ncheckpoint: {SENTENCE!r} -> {tok.encode(SENTENCE)}")
