"""Listing 1.1: a language model in about twenty lines, by counting pairs of words.
Run from 04-code/ch01:  python3 bigram.py"""
import sys, os, random
from collections import Counter, defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tiny.corpus import load_text, words

SEED = 1865                                    # printed so you can reproduce every line
random.seed(SEED)

tokens = words(load_text())
pairs = defaultdict(Counter)                   # pairs["the"]["queen"] = how often "queen" follows "the"
for a, b in zip(tokens, tokens[1:]):
    pairs[a][b] += 1

def next_word_shares(word):
    """Turn counts into shares that add up to one."""
    counts = pairs[word]
    total = sum(counts.values())
    return {w: c / total for w, c in counts.most_common()}

def generate(start, n=12):
    out = [start]
    for _ in range(n):
        shares = next_word_shares(out[-1])
        if not shares: break
        out.append(random.choices(list(shares), weights=list(shares.values()))[0])
    return " ".join(out)

if __name__ == "__main__":
    print(f"seed = {SEED}")
    print(f"corpus: {len(tokens):,} words, {len(set(tokens)):,} different words")
    print(f'"the" appears {tokens.count("the"):,} times and is followed by {len(pairs["the"]):,} different words')
    print('\nTop five next words after "the":')
    for w, s in list(next_word_shares("the").items())[:5]:
        print(f"  {w:<10} {pairs['the'][w]:>4} times   share {s:.3f}")
    for start in ["the", "alice", "the"]:
        print(f"\ngenerated: {generate(start)}")
