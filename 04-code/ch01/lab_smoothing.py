"""Lab 1: score sentences with the pair-counting model, then break it by removing smoothing.
Run from 04-code/ch01:  python3 lab_smoothing.py"""
import sys, os
from collections import Counter, defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tiny.corpus import load_text, words

tokens = words(load_text())
cut = int(len(tokens) * 0.9)                       # first 90% to count, last 10% to test
train, test = tokens[:cut], tokens[cut:]
vocab = sorted(set(tokens))
V = len(vocab)

def count_pairs(seq):
    pairs = defaultdict(Counter)
    for a, b in zip(seq, seq[1:]): pairs[a][b] += 1
    return pairs

pairs = count_pairs(train)

def share(a, b, add=0.0):
    """Share of the times word a is followed by word b. add=1 is add-one smoothing:
    pretend every possible pair was seen once more than it was."""
    total = sum(pairs[a].values())
    return (pairs[a][b] + add) / (total + add * V)

def sentence_score(sentence, add):
    ws = sentence.split()
    p = 1.0
    rows = []
    for a, b in zip(ws, ws[1:]):
        s = share(a, b, add); p *= s
        rows.append((a, b, pairs[a][b], s))
    return p, rows

if __name__ == "__main__":
    print(f"counted on the first {len(train):,} words; tested on the last {len(test):,}; vocabulary {V:,} words\n")
    for add, name in [(1.0, "WITH add-one smoothing"), (0.0, "BROKEN: smoothing removed")]:
        p, rows = sentence_score("the cat sat on the mat", add)
        print(f'{name}:  "the cat sat on the mat"')
        for a, b, c, s in rows:
            print(f"   {a:>4} -> {b:<4} seen {c:>3} times   share {s:.6f}")
        print(f"   sentence probability = {p:.3e}\n")
    test_pairs = list(zip(test, test[1:]))
    unseen = sum(1 for a, b in test_pairs if pairs[a][b] == 0)
    print(f"pairs in the test text: {len(test_pairs):,}")
    print(f"pairs never seen while counting: {unseen:,} ({unseen/len(test_pairs):.1%})")
    zero_sents = 0; sents = 0
    for i in range(0, len(test) - 6, 6):           # every six-word stretch of the test text
        sents += 1
        if any(pairs[a][b] == 0 for a, b in zip(test[i:i+6], test[i+1:i+6])): zero_sents += 1
    print(f"six-word stretches of real text the unsmoothed model calls impossible: {zero_sents:,} of {sents:,} ({zero_sents/sents:.1%})")
    print("\nhow the share of never-seen test pairs falls as more text is counted:")
    for f in [0.1, 0.3, 0.5, 0.7, 0.9]:
        tr = tokens[:int(len(tokens) * f)]; seen = set(zip(tr, tr[1:]))
        print(f"  {len(tr):>6,} words counted -> {sum(p not in seen for p in test_pairs)/len(test_pairs):.1%} of test pairs unseen")
