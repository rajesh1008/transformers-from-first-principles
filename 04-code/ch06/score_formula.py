"""Why only the gap matters: turning both arrows = turning one arrow by the gap.  Checked on every pair of words
of the toy vocabulary and 50 position pairs each, then the two-part formula.  Run: python3 score_formula.py"""
import _common, math
import numpy as np
from tiny.tokenizer import TOY_VOCAB
from tiny.embed import TOY_EMBED, rotate_pair, dot

rng = np.random.default_rng(1865)
worst = 0.0
for a in TOY_EMBED:
    for b in TOY_EMBED:
        for m, n in rng.integers(0, 1000, size=(50, 2)):
            both = dot(rotate_pair(*a, m * math.radians(30)), rotate_pair(*b, n * math.radians(30)))
            one = dot(a, rotate_pair(*b, (n - m) * math.radians(30)))
            worst = max(worst, abs(both - one))
print(f"{len(TOY_EMBED) ** 2} word pairs x 50 position pairs: largest difference, both turned vs one turned by the gap: {worst:.1e}")
(a, b), (c, d) = TOY_EMBED[TOY_VOCAB.index("sat")], TOY_EMBED[TOY_VOCAB.index("on")]
print(f"sat = ({a:g}, {b:g}), on = ({c:g}, {d:g}): straight part ac+bd = {a * c + b * d:g}, crossed part ad-bc = {a * d - b * c:g}")
for gap in [0, 1, 2, 3, 6]:
    t = gap * math.radians(30)
    formula = (a * c + b * d) * math.cos(t) + (a * d - b * c) * math.sin(t)
    direct = dot((a, b), rotate_pair(c, d, t))
    print(f"  gap {gap}: formula {round(formula, 9) + 0.0:6.3f}   turning on directly {round(direct, 9) + 0.0:6.3f}")
for gap in [1, -1, 3, -3]:
    print(f"  sat asking, on answering at gap {gap:+d}: {round(dot((a, b), rotate_pair(c, d, gap * math.radians(30))), 9) + 0.0:6.3f}")
