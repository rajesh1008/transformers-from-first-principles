"""Numbers used in the Appendix A solutions for Chapter 1.  Run from 04-code/ch01."""
import sys, os
from collections import Counter, defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tiny.corpus import load_text, words
tokens = words(load_text())
pairs = defaultdict(Counter)
for a, b in zip(tokens, tokens[1:]): pairs[a][b] += 1
V = len(set(tokens))
one = sum(1 for c in pairs.values() if len(c) == 1)
print(f"Ex 6: {one:,} of {V:,} words ({one/V:.1%}) are always followed by the same word")
train = tokens[:int(len(tokens) * 0.9)]
on_total = sum(1 for a in train[:-1] if a == "on")
print(f"Ex 4: 'on' is followed by a word {on_total} times in the counted text; 29/{on_total} = {29/on_total:.6f}; 30/{on_total + V} = {30/(on_total + V):.6f}")
print(f"Ex 12: four-token grid for GPT-2's vocabulary: {50_257**4:,} cells")
