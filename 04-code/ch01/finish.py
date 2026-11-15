"""Finish this sentence: "The rabbit looked at ..." -- what the corpus says comes after "at".
Run from 04-code/ch01:  python3 finish.py"""
import sys, os
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tiny.corpus import load_text, words

tokens = words(load_text())
after_at = Counter(b for a, b in zip(tokens, tokens[1:]) if a == "at")
total = sum(after_at.values())
print(f'"at" is followed by a word {total} times, by {len(after_at)} different words')
for w, n in after_at.most_common(6):
    print(f"  at {w:<6} {n:>3} times  {n/total:.1%}")
