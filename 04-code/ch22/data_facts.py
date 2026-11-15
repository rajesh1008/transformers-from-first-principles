"""Chapter 22, the new data: sizes of the five Carroll books, the pool, contamination checks of the pool against both
held-out sets (13-word runs, as in s22-4), and how many windows the pool gives."""
import _common
import numpy as np
from tiny import data as D
from _pool import corpora, pool_windows, POOL_BOOKS, HOLD, book_text
from _common import save_json
c = corpora(); T = c["T"]
out = {}
print("Part 1. the Carroll pool")
for b in POOL_BOOKS:
    txt = book_text(b)
    print(f"  {b:15s} {len(txt.split()):7,} words {len(c['books'][b]):8,} tokens  {D.SOURCES[b][1]}")
    out[b] = len(c["books"][b])
w, _ = pool_windows()
print(f"  pool {len(c['pool']):,} tokens -> {len(w):,} windows of 129 (128 predictions each); held out: last {HOLD:,} tokens of sylvie2")
print(f"  a 400-step run of 8 windows needs 3,200 windows: the pool gives {len(w) / 3200:.3f} passes' worth")
out["pool"] = len(c["pool"]); out["windows"] = len(w)
print("\nPart 2. contamination: distinct 13-word runs of each held-out text also found in the training text")
pool_text = T.decode(c["pool"].tolist())          # the pool as trained on (first version of this check used the whole books: see RUN.txt)
whole_text = "\n".join(D.load_source(b) for b in POOL_BOOKS)
held_b_text = T.decode(c["held_b"].tolist())
for name, test in [("Looking-Glass", D.load_source("glass")), ("Carroll held-out", held_b_text)]:
    for tr_name, tr in [("pool", pool_text), ("Alice", D.load_source("alice")), ("whole books", whole_text)]:
        k, n = D.ngram_overlap(tr, test)
        print(f"  {name:17s} vs {tr_name:6s}: {k:,} of {n:,}")
        out[f"{name}|{tr_name}"] = [k, n]
save_json("data_facts", out)
