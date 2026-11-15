"""Where This Bites in Production: a model that only knows "what usually comes next"
is confidently wrong whenever the usual continuation is not the true one.
Run from 04-code/ch01:  python3 confident_wrong.py"""
import sys, os
from collections import Counter, defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tiny.corpus import load_text, words

tokens = words(load_text())
pairs = defaultdict(Counter)
for a, b in zip(tokens, tokens[1:]): pairs[a][b] += 1

def top(word, k=3):
    c = pairs[word]; total = sum(c.values())
    return [(w, n / total) for w, n in c.most_common(k)]

for prompt in ["the capital of france is", "the mock", "the march", "said the"]:
    last = prompt.split()[-1]
    guesses = ", ".join(f"{w} ({s:.0%})" for w, s in top(last))
    print(f'{prompt!r:<28} -> {guesses}')
