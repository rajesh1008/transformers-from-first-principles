"""Chapter 10, section 3: grouped-query attention by hand on four heads.  Lab 9's four seeded heads
(seed 1865) on the running sentence keep their own queries; their keys and values are shared:
multi-head (each head its own), grouped-query (heads 0 and 1 use head 0's keys and values, heads 2 and 3
use head 2's), multi-query (every head uses head 0's).  Run from 04-code/ch10:  python3 gqa_by_hand.py"""
from _common import *
from tiny.attention import attention

rng = np.random.default_rng(SEED)
heads = [tuple(np.round(rng.normal(size=(2, 2)), 2) for _ in range(3)) for _ in range(4)]
W_o = np.round(rng.normal(size=(8, 2)), 2)
R = XP
variants = {"multi-head": [0, 1, 2, 3], "grouped-query (2 groups)": [0, 0, 2, 2], "multi-query (1 group)": [0, 0, 0, 0]}
full = None
for name, kv in variants.items():
    outs, tabs = [], []
    for h in range(4):
        q = heads[h][0]; k, v = heads[kv[h]][1], heads[kv[h]][2]
        o, A = attention(R, q, k, v)
        outs.append(o); tabs.append(A)
    final = np.concatenate(outs, axis=1) @ W_o
    cached = len(set(kv)) * 2 * 2                      # key + value, 2 numbers each, per distinct kv head
    print(f"\n{name}: key-value heads used {kv}; numbers cached per token: {cached}")
    for h in range(4):
        print(f"  head {h}: sat's row {r3(tabs[h][2, :3])}   mat's row {r3(tabs[h][5])}")
    print("  sat's final output", r3(final[2]), " mat's final output", r3(final[5]))
    if full is None:
        full = final; full_tabs = tabs
    else:
        print(f"  largest change in any share from multi-head: {max(np.abs(a - b).max() for a, b in zip(tabs, full_tabs)):.3f}"
              f" | largest change in a final output number: {np.abs(final - full).max():.3f}")
        same = [h for h in range(4) if np.allclose(tabs[h], full_tabs[h])]
        print("  heads whose tables are unchanged:", same)
