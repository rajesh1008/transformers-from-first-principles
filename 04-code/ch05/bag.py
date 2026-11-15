"""Section 5.1: without positions, a sentence is a bag of rows.  Run: python3 bag.py"""
import _common
from _common import SENTENCE, SCRAMBLED
from itertools import permutations
from tiny.tokenizer import TOY_VOCAB
from tiny.embed import TOY_EMBED, dot

def rows(text):
    return [TOY_EMBED[TOY_VOCAB.index(w)].tolist() for w in text.split()]

A, B = rows(SENTENCE), rows(SCRAMBLED)
print("rows for", repr(SENTENCE), A)
print("rows for", repr(SCRAMBLED), B)
print("same rows, ignoring order (same bag):", sorted(A) == sorted(B))

def average(rs):
    return [sum(r[i] for r in rs) / len(rs) for i in range(2)]
print("average row, first sentence: ", [round(x, 3) for x in average(A)])
print("average row, second sentence:", [round(x, 3) for x in average(B)])

# What 'sat' sees: its agreement (dot product) with every word of the sentence
for text, rs in ((SENTENCE, A), (SCRAMBLED, B)):
    words = text.split()
    sat = rs[words.index("sat")]
    view = sorted((w, dot(sat, r)) for w, r in zip(words, rs))
    print(f"'sat' agrees with, in {text!r}:", [(w, int(v)) for w, v in view])

words = SENTENCE.split()
orders = set(permutations(words))
print("different orders of these six words:", len(orders), "(6 x 5 x 4 x 3 x 2 x 1 = 720, halved because 'the' appears twice)")
print("orders that give exactly the same bag of rows:", sum(1 for o in orders if sorted(rows(' '.join(o))) == sorted(A)))
