"""Numerical answers to the Chapter 6 exercises (Appendix A).  Run: python3 solutions.py"""
import _common, math
import numpy as np
from _common import SENTENCE
from tiny.tokenizer import TOY_VOCAB
from tiny.embed import (TOY_EMBED, TOY_TURN, rotate_pair, apply_rotary, agreement_table, dot, length,
                        clock, clock_speeds, add_positions)

ROW = {w: TOY_EMBED[TOY_VOCAB.index(w)] for w in TOY_VOCAB}
def turned(w, pos):
    return apply_rotary([ROW[w]], [pos], speeds=TOY_TURN)[0]
print("1.", "dog@3", turned("dog", 3).round(3).tolist(), "hat@2", turned("hat", 2).round(3).tolist(),
      "lengths", round(length(turned("dog", 3)), 3), round(length(turned("hat", 2)), 3))
x, y = rotate_pair(1, 1, math.radians(60))
print("2.", "the@0.sat@2 =", round(dot(turned("the", 0), turned("sat", 2)), 3), "; the.(sat turned 60) =", round(dot((1, 0), (x, y)), 3))
print("3.", "the@7.sat@9 =", round(dot(turned("the", 7), turned("sat", 9)), 3), "; sat@9.the@7 =", round(dot(turned("sat", 9), turned("the", 7)), 3))
target = tuple(turned("the", 0).round(6))
hits = [(w, p) for w in TOY_VOCAB[:-1] for p in range(12) if tuple(turned(w, p).round(6)) == target]
print("4.", "mat@5 =", turned("mat", 5).round(3).tolist(), "sat@2 =", turned("sat", 2).round(3).tolist(),
      "; pairs landing on the@0's arrow:", hits)
reps = [g for g in range(1, 30) if abs(dot(turned("the", 0), turned("cat", g)) - dot(turned("the", 0), turned("cat", 1))) < 1e-9 and g != 1]
print("5.", "gaps with the same the-cat agreement as gap 1:", reps[:3])
print("6.", "angles per 1,024-token window:", 1024 * 64)
for base in [10000, 500000]:
    sp = clock_speeds(128, base)
    print("7.", f"base {base:,}: fastest turn every {2 * math.pi / sp[0]:.2f} tokens, slowest every {2 * math.pi / sp[-1]:,.0f}")
f = 1024 / 8192
print("8.", f"squeezed fastest clock: {math.degrees(f):.2f} deg/token; neighbours agree {math.cos(f):.4f}")
X = TOY_EMBED[[TOY_VOCAB.index(w) for w in SENTENCE.split()]]
t0 = agreement_table(add_positions(X, [clock(k) for k in range(6)]))
t7 = agreement_table(add_positions(X, [clock(k) for k in range(7, 13)]))
print("9.", "added clock rows, agreements that change when moved to 7-12:", int((np.abs(t0 - t7) > 1e-9).sum()))
rng = np.random.default_rng(1865)
q, k = rng.normal(size=128), rng.normal(size=128)
sp16 = clock_speeds(128)[:8]                          # turn only the first 16 numbers, with the 8 fastest clocks
def partial(v, pos):
    out = v.copy(); out[:16] = apply_rotary([v[:16]], [pos], speeds=sp16)[0]; return out
print("11.", "partial rotary shift test (0,5) vs (300,305):",
      round(dot(partial(q, 0), partial(k, 5)), 6), round(dot(partial(q, 300), partial(k, 305)), 6),
      "; full-row dot", round(dot(q, k), 3), "; turned part at gap 0 and 5:",
      round(dot(q[:16], k[:16]), 3), round(dot(apply_rotary([q[:16]], [0], speeds=sp16)[0], apply_rotary([k[:16]], [5], speeds=sp16)[0]), 3))
full = clock_speeds(128)
Q = rng.normal(size=(300, 128)); K = rng.normal(size=(300, 128))
for gap in [1, 10, 100]:
    d = np.mean([abs(dot(apply_rotary([a], [0], speeds=full)[0], apply_rotary([b], [gap / 2], speeds=full)[0])
                     - dot(apply_rotary([a], [0], speeds=full)[0], apply_rotary([b], [gap], speeds=full)[0])) for a, b in zip(Q, K)])
    print("12.", f"squeezing by 2 at serving, gap {gap}: average change {d:.2f}")
from bites import bf16
kept = sorted(set(float(bf16(p)) for p in range(1000, 1024)))
print("13.", f"positions 1,000-1,023 stored as bfloat16: {len(kept)} distinct values survive: {[int(v) for v in kept]};",
      f"float16 keeps {len(set(float(np.float16(p)) for p in range(1000, 1024)))}")
