"""Chapter 8: why divide by the square root of the key width.  Random query and key rows whose
numbers have spread 1 (what Chapter 13's normalisation produces), seed 1865.
Run from 04-code/ch08:  python3 saturation.py"""
from _common import *
from tiny.attention import softmax, entropy

rng = np.random.default_rng(1865)
n = 6
print(f"rows of {n} keys, 2,000 random queries per width; numbers drawn with spread 1, seed 1865")
print(" width | score spread | unscaled: top share  tokens>=1%  entropy | scaled: top share  tokens>=1%  entropy")
for d in [2, 16, 64, 128, 512, 1024]:
    q = rng.normal(size=(2000, d)); k = rng.normal(size=(2000, n, d))
    s = np.einsum("bd,bnd->bn", q, k)
    a_raw, a_sc = softmax(s), softmax(s / np.sqrt(d))
    def stats(a):
        return a.max(axis=1).mean(), (a >= 0.01).sum(axis=1).mean(), entropy(a).mean()
    r, c = stats(a_raw), stats(a_sc)
    print(f" {d:>5} | {s.std():12.2f} | {r[0]:19.3f} {r[1]:11.2f} {r[2]:8.3f} | {c[0]:17.3f} {c[1]:11.2f} {c[2]:8.3f}")
print(f" (a row of {n} even shares has entropy ln {n} = {np.log(n):.3f}; all on one token, 0)")

# how much a share can still move: raise the top score by 0.1 and watch the top share
print("\nhow far the top share moves when its score rises by 0.1 (width 128, 6 keys, 2,000 rows):")
d = 128
q = rng.normal(size=(2000, d)); k = rng.normal(size=(2000, n, d))
s = np.einsum("bd,bnd->bn", q, k)
for label, rows in [("unscaled", s), ("scaled", s / np.sqrt(d))]:
    a = softmax(rows); j = a.argmax(axis=1)
    bumped = rows.copy(); bumped[np.arange(len(rows)), j] += 0.1
    moved = softmax(bumped)[np.arange(len(rows)), j] - a[np.arange(len(rows)), j]
    stuck = (a.max(axis=1) > 0.99).mean()
    print(f"  {label:>8}: average move {moved.mean():.4f}; rows with a top share above 0.99: {stuck:.1%}; median move {np.median(moved):.5f}")

print("\none window of Tiny's length, 1,024 keys, width 128:")
q = rng.normal(size=(200, 128)); k = rng.normal(size=(200, 1024, 128))
s = np.einsum("bd,bnd->bn", q, k)
for label, row in [("unscaled", s), ("scaled", s / np.sqrt(128))]:
    a = softmax(row)
    print(f"  {label:>8}: average top share {a.max(axis=1).mean():.3f}; tokens holding 90% of the share {np.mean([np.searchsorted(np.cumsum(np.sort(x)[::-1]), 0.9) + 1 for x in a]):.1f} of 1,024")
