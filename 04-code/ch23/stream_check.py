"""How well each shard layout mixes the books: the first ladder's four whole-book shards against the 24 shards of
shuffled 4,096-token pieces.  For each, the largest single-book share in a batch of eight (first 48 batches) and the
book read in the last sixth of a one-pass run (steps 320 to 383) for the three reading orders."""
import _common
import os, numpy as np
from collections import Counter
from tiny import data as D
from _pool import corpora, POOL_BOOKS, SHARDS
c = corpora()
pool = c["pool"]; starts = np.cumsum([0] + [len(c["books"][b]) for b in POOL_BOOKS])
idx = {tuple(pool[i:i + 16]): i for i in range(0, len(pool) - 16)}
def book(w):
    return int(np.searchsorted(starts, idx.get(tuple(w[:16]), -1), side="right") - 1)
for name, d in (("four whole-book shards (first ladder)", os.path.join(_common.HERE, "outputs", "v1_four_shards", "shards")),
                ("24 shards of shuffled pieces", SHARDS)):
    print(name)
    for seed in (1865, 7, 11):
        st = D.ShardStream(d, seed=seed)
        bs = [st.next_batch() for _ in range(384)]
        share = np.mean([max(Counter(book(w) for w in b).values()) / 8 for b in bs[:48]])
        last = Counter(book(w) for b in bs[320:] for w in b)
        top = last.most_common(1)[0]
        print(f"  order {seed:4d}: largest single-book share in a batch {share:.2f}; steps 320-383 read {top[1] / sum(last.values()):.0%} from {POOL_BOOKS[top[0]]}")
