"""BPE training in its most obvious form: recount every pair before every merge.
Slow, but easy to check by eye.  Tiny's train_bpe gives the same merges, much faster.
Run from 04-code/ch03:  python3 bpe_simple.py"""
import _common, time
from collections import Counter
from tiny.corpus import load_text
from tiny.tokenizer import chunks, merge_pair, train_bpe

def train_simple(text, n_merges):
    counts = Counter(chunks(text))                       # each different chunk, and how often it occurs
    words = {c: list(c.encode("utf-8")) for c in counts}  # each chunk as a list of byte ids
    merges = []
    for new_id in range(256, 256 + n_merges):
        pairs = Counter()
        for c, w in words.items():                       # count every neighbouring pair
            for p in zip(w, w[1:]):
                pairs[p] += counts[c]
        best = max(pairs, key=lambda p: (pairs[p], -p[0], -p[1]))   # most frequent; ties to smaller ids
        merges.append((best, new_id, pairs[best]))
        words = {c: merge_pair(w, best, new_id) for c, w in words.items()}
    return merges

text = load_text()
t0 = time.perf_counter(); slow = train_simple(text, 500); t_slow = time.perf_counter() - t0
t0 = time.perf_counter(); fast = train_bpe(text, 256 + 500); t_fast = time.perf_counter() - t0
print(f"simple version: 500 merges in {t_slow:.2f} s")
print(f"Tiny's version: 500 merges in {t_fast:.3f} s")
print("the two lists of merges are identical:", slow == fast)
