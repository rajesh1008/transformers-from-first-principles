"""Write the Carroll pool as uint16 shards once, before parallel runs start, and print what ShardStream serves."""
import _common
import numpy as np
from _pool import shards, corpora
from tiny import data as D
d = shards()
idx, maps = D.open_shards(d)
print(f"{len(idx)} shards, {sum(s['tokens'] for s in idx):,d} tokens; pool {len(corpora()['pool']):,d} tokens")
st = D.ShardStream(d, seed=1865)
n = 0
while st.passes == 0:
    st._next_window(); n += 1
print(f"windows of 129 in one pass of the stream: {n:,d}  (= {n * 128:,d} predictions)")
from _pool import pieces, POOL_BOOKS
print(f"pieces of up to 4,096 tokens: {len(pieces())}")
st = D.ShardStream(d, seed=1865)
book_of = []
c = corpora()
starts = np.cumsum([0] + [len(c["books"][b]) for b in POOL_BOOKS])
from collections import Counter
# which book each served window came from, for the first 48 batches, by matching its first 16 tokens in the pool
pool = c["pool"]
idx = {tuple(pool[i:i + 16]): i for i in range(0, len(pool) - 16)}
share = []
for _ in range(48):
    bt = st.next_batch()
    bk = [int(np.searchsorted(starts, idx.get(tuple(w[:16]), -1), side="right") - 1) for w in bt]
    share.append(max(Counter(bk).values()) / len(bk))
print(f"largest single-book share in a batch of 8, first 48 batches: mean {np.mean(share):.2f}")
