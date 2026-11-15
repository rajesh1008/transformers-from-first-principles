"""Chapter 8 scoreboard.  Run from 04-code/ch08:  python3 scoreboard.py  (--measure to time yours)"""
from _common import *
import time
from tiny.attention import Head
from tiny.embed import Embedding, PositionTable
from tiny.tokenizer import load_bpe
from tiny.corpus import load_text

tok = load_bpe(TINY_BPE)
emb, pos, head = Embedding(len(tok)), PositionTable(), Head()
ids = np.array(tok.encode(load_text()))
windows = [ids[i:i + 1024] for i in range(0, len(ids), 1024)]
def measure():
    t = time.perf_counter()
    for w in windows:
        head(emb(w) + pos(range(len(w))))
    ms = (time.perf_counter() - t) * 1000
    return {"ms": ms, "tokens": int(len(ids))}
r = recorded("scoreboard", measure)
print(f"one head over the whole novel ({r['tokens']:,} tokens, {len(windows)} windows): {r['ms']:.0f} ms ({r['tokens'] / r['ms'] * 1000:,.0f} tokens per second)  (measured {r['measured']})")
learned = emb.table.size + pos.table.size + head.n_params
print(f"learned numbers: 262,144 (tokens, tied) + 131,072 (positions) + {head.n_params:,} (one head: query, key, value) = {learned:,}")
print("the score table for one full window: 1,024 x 1,024 =", f"{1024 * 1024:,}", "shares, about half of them masked to zero")
