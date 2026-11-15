"""Chapter 22, s22-9: the unigram-surprise score of the 3,072 windows of the 1-pass run (the same windows
train_data.py orders for its curricula): range, which books fill the easiest and hardest tenths, and the two ends."""
import _common
import numpy as np
from tiny import data as D
from _pool import corpora, pool_windows, POOL_BOOKS
from _common import SEED, save_json
c = corpora(); W, B = pool_windows()
perm = np.random.default_rng(SEED + 22).permutation(len(W)); idx = np.sort(perm[:3072])
sl, bk = W[idx], B[idx, 0]
u = D.unigram_surprise(sl, np.bincount(c["pool"], minlength=2048).astype(float))
o = np.argsort(u, kind="stable")
print(f"unigram surprise of the 3,072 windows: min {u.min():.3f}, median {np.median(u):.3f}, max {u.max():.3f}")
for name, part in [("all windows", np.arange(len(u))), ("easiest tenth", o[:307]), ("hardest tenth", o[-307:])]:
    print(f"  {name:14s} " + ", ".join(f"{b} {int((bk[part] == k).sum())}" for k, b in enumerate(POOL_BOOKS)))
T = c["T"]
for name, j in [("EASIEST", o[0]), ("HARDEST", o[-1])]:
    print(f"{name}: {T.decode(sl[j].tolist())[:150]!r}")
save_json("curriculum_scores", dict(min=float(u.min()), median=float(np.median(u)), max=float(u.max())))
