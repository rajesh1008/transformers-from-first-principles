"""Chapter 10, section 9: Tiny's variants with tiny/attention.py's new options.  Checks that the defaults
reproduce Chapter 9, counts learned numbers and cache per variant, and times each on the novel
(fastest of five, recorded once; --measure to time your machine).  Run:  python3 tiny_variants.py"""
from _common import *
import time
from tiny.attention import MultiHead
from tiny.embed import Embedding, PositionTable
from tiny.tokenizer import load_bpe
from tiny.corpus import load_text

tok = load_bpe(TINY_BPE)
ids = tok.encode(SENTENCE)
Xs = Embedding(len(tok))(ids) + PositionTable()(range(len(ids)))
base = MultiHead(n_heads=4)
same = MultiHead(n_heads=4, n_kv_heads=4, window=None)
print("defaults reproduce Chapter 9 exactly:", bool(np.array_equal(base(Xs), same(Xs))))
wide = MultiHead(n_heads=4, window=len(ids))
print("a window as long as the text changes nothing:", bool(np.allclose(base(Xs), wide(Xs))))
g2 = MultiHead(n_heads=4, n_kv_heads=2)
print("grouped-query keeps the first two key blocks of the same seeded table:",
      bool(np.array_equal(g2.W_k, base.W_k[:, :64])))

text = load_text()
all_ids = np.array(tok.encode(text))
n = 1024
Xn = Embedding(len(tok))(all_ids[:n]) + PositionTable()(range(n))
variants = [("multi-head, 4 kv heads", dict(n_heads=4)),
            ("grouped-query, 2 kv heads", dict(n_heads=4, n_kv_heads=2)),
            ("multi-query, 1 kv head", dict(n_heads=4, n_kv_heads=1)),
            ("multi-head, window 256", dict(n_heads=4, window=256)),
            ("grouped-query 2 + window 256", dict(n_heads=4, n_kv_heads=2, window=256))]
def measure():
    res = {}
    for name, kw in variants:
        m = MultiHead(**kw)
        best = 1e9
        for _ in range(5):
            t = time.perf_counter(); m(Xn); best = min(best, time.perf_counter() - t)
        res[name] = round(best * 1000, 1)
    return res
times = recorded("tiny_variants_timing", measure)
print(f"\nTiny's layer on the first {n:,} tokens of the novel ({len(all_ids):,} tokens in all):")
print(f"{'variant':<30}{'learned numbers':>16}{'cache per token':>17}{'cache at 1,024':>16}{'ms (fastest of 5)':>19}")
for name, kw in variants:
    m = MultiHead(**kw)
    per_tok = 2 * m.n_kv_heads * m.d_head
    kept = min(n, kw.get("window") or n)
    print(f"{name:<30}{m.n_params:>16,}{per_tok:>17}{per_tok * kept:>16,}{times[name]:>19}")
print("(timings measured", times["measured"], "on the book's build machine; NumPy computes the whole table and masks it,")
print(" so a window saves memory in a real cache but no time here)")
m = MultiHead(n_heads=4, window=256); _, A = m(Xn, return_shares=True)
print("\nwindow 256 on 1,024 tokens: rows sum to one:", bool(np.allclose(A.sum(-1), 1)))
far = np.tril(np.ones((n, n), bool), -256)
print("shares outside the window, total:", float(A[:, far].sum()))
