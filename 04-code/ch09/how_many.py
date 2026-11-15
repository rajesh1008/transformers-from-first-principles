"""Chapter 9: how many heads?  Tiny's width of 128 cut into 1, 2, 4, ... 128 heads (all timed; 128 heads needs about 4 GB of memory free).  The learned
numbers do not change (65,536 with the output table); the number of shares to compute and hold
grows with the number of heads; the time is measured on one full window of the novel (1,024 rows).
Then the limit of the narrowest head: a head of width 1 ranks every earlier key in the same order
(or exactly reversed) for every query, so its favourite is always the largest or smallest key so far.
Run from 04-code/ch09:  python3 how_many.py   (add --measure to time your own machine)"""
from _common import *
import time
from tiny.attention import MultiHead, attention
from tiny.embed import Embedding, PositionTable
from tiny.tokenizer import load_bpe
from tiny.corpus import load_text

tok = load_bpe(TINY_BPE)
novel = np.array(tok.encode(load_text()))
emb, pos = Embedding(len(tok)), PositionTable()
W = emb(novel[:1024]) + pos(range(1024))
counts = [1, 2, 4, 8, 16, 32, 64, 128]          # 64 and 128 heads hold 0.5 and 1 GB of shares per window in float64
all_counts = [1, 2, 4, 8, 16, 32, 64, 128]
def measure():
    res = {}
    for h in counts:
        m = MultiHead(n_heads=h)
        m(W)                                         # warm up
        best = float("inf")
        for _ in range(5):                           # the fastest of five runs: the least disturbed by other work
            t = time.perf_counter(); m(W)
            best = min(best, time.perf_counter() - t)
        res[str(h)] = best * 1000
    return res
r = recorded("how_many_timing", measure)
print("heads  width  learned numbers  shares per window  time per window (ms)")
for h in all_counts:
    m = MultiHead(n_heads=h)
    t = f"{r[str(h)]:>10.1f}" if str(h) in r else "  not timed"
    print(f"{h:>5}  {128 // h:>5}  {m.n_params:>15,}  {h * 1024 * 1024:>17,}  {t}  ({h * 1024 * 1024 * 8 / 2**20:,.0f} MB of shares in float64)")
print("measured:", r["measured"], "(book build machine, CPU; fastest of five runs)")

# the narrowest head: width 1, random tables, rows with spread 1 so that the shares are not all even
rng = np.random.default_rng(SEED)
R = rng.normal(size=(1024, 128))
wq, wk, wv = (rng.normal(size=(128, 1)) / np.sqrt(128) for _ in range(3))
_, A = attention(R, wq, wk, wv)
k = (R @ wk)[:, 0]; q = (R @ wq)[:, 0]
fav = A.argmax(axis=1)
run_max = np.array([np.argmax(k[:i + 1]) for i in range(1024)]); run_min = np.array([np.argmin(k[:i + 1]) for i in range(1024)])
ok = np.where(q > 0, fav == run_max, fav == run_min)
print(f"\nwidth-1 head on 1,024 random rows (seed {SEED}): rows whose favourite is the largest key so far (query > 0)"
      f" or the smallest (query < 0): {ok.sum()} of 1024")
print(f"distinct favourites across all 1,024 rows: {len(set(fav.tolist()))}")
B = MultiHead(n_heads=4)
R32 = R
_, A4 = B(R32 * 25, return_shares=True)              # large rows so that random heads of 32 are not averagers
print(f"for comparison, four heads of width 32 on the same rows scaled up: distinct favourites per head:",
      [len(set(a.argmax(1).tolist())) for a in A4])
