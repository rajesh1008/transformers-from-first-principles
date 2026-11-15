"""A lookup table: token ids become rows of numbers.
Run from 04-code/ch04:  python3 lookup.py"""
import _common
from _common import SENTENCE
from tiny.tokenizer import WordTokenizer, TOY_VOCAB
from tiny.embed import TOY_EMBED

tok = WordTokenizer(TOY_VOCAB)
ids = tok.encode(SENTENCE)                     # Chapter 2: [0, 1, 2, 3, 0, 4]
print("the toy table: one row of two numbers per vocabulary entry")
for i, (w, row) in enumerate(zip(tok.itos, TOY_EMBED)):
    print(f"  row {i:>2}  {w:<7} ({row[0]:>2.0f}, {row[1]:>2.0f})")
print(f"\n{SENTENCE!r} -> ids {ids}")
X = TOY_EMBED[ids]                            # the whole lookup: one row per id, in order
print("looked up, one row per token:")
for w, i, row in zip(SENTENCE.split(), ids, X):
    print(f"  {w:<4} id {i}  ->  ({row[0]:>2.0f}, {row[1]:>2.0f})")
print(f"shape of the result: {X.shape[0]} rows of {X.shape[1]} numbers")
print(f"rows for position 0 and position 4 identical: {(X[0] == X[4]).all()}")
