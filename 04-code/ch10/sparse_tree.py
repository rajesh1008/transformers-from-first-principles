"""Chapter 10, section 7: indexed (tree) sparse attention, a toy.  Group every C keys under a parent
with a mean-pooled key and value; let each query score the parents, open only the best k, attend exactly
to the tokens inside them and to the parents' summaries for the rest.  Counts entries read and measures
the error against full attention on one synthetic problem.  The 2026 method this imitates learns its
pooled keys; this toy uses plain means.  Run:  python3 sparse_tree.py"""
from _common import *
from tiny.attention import softmax

rng = np.random.default_rng(SEED)
n, d, C = 4096, 32, 16
K = rng.normal(size=(n, d)); V = rng.normal(size=(n, d))
# plant 8 "needles": keys the query should find
needles = rng.choice(n - 64, size=8, replace=False)
q = rng.normal(size=d) * 0.3
for j in needles:
    K[j] = q / np.linalg.norm(q) * 6 + rng.normal(size=d) * 0.1
scale = 1 / np.sqrt(d)
full = softmax(K @ q * scale) @ V
Kp = K.reshape(n // C, C, d).mean(1); Vp = V.reshape(n // C, C, d).mean(1)
print(f"n={n:,} keys of width {d}, groups of C={C} -> {n // C} parents; seed {SEED}")
print("open k parents | entries read | share of the {:,} keys | error vs full (relative)".format(n))
for k in (4, 8, 16, 32, 64, 256):
    sp = Kp @ q * scale
    top = np.argsort(-sp)[:k]
    rest = np.setdiff1d(np.arange(n // C), top)
    idx = np.concatenate([np.arange(t * C, (t + 1) * C) for t in top])
    scores = np.concatenate([K[idx] @ q * scale, Kp[rest] @ q * scale + np.log(C)])   # a summary stands for C tokens
    vals = np.concatenate([V[idx], Vp[rest]])
    out = softmax(scores) @ vals
    read = n // C + len(idx)
    err = np.linalg.norm(out - full) / np.linalg.norm(full)
    found = len(set(needles) & set(idx.tolist()))
    print(f"  {k:>4}           | {read:>6,}       | {read / n:6.1%}            | {err:.3f}   needles opened {found}/8")
print("\ntwo levels of 16 (256-token grandparents), opening 8 at each level:")
G = 16
Kg = Kp.reshape(-1, G, d).mean(1)
tg = np.argsort(-(Kg @ q))[:8]
cand = np.concatenate([np.arange(t * G, (t + 1) * G) for t in tg])
tp = cand[np.argsort(-(Kp[cand] @ q))[:8]]
read2 = len(Kg) + len(cand) + 8 * C
print(f"  entries scored or read: {len(Kg)} + {len(cand)} + {8 * C} = {read2} of {n:,};"
      f" needles inside opened parents: {len(set(needles) & set(np.concatenate([np.arange(t*C,(t+1)*C) for t in tp]).tolist()))}/8")
