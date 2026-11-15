"""Chapter 40, 40.6 (second half): is the direction a probe finds also a handle?  For each held-out animal A (verb s), choose a target verb t != s,
build the direction d = mean row at the site (layer l) of the other animals taught t minus the mean row of the other animals taught s, add a * d
to A's row at the site at layer l, and let the fine-tuned Tiny finish.  Share of the 30 animals whose first prediction becomes t, whose stays s.
a = 1 (the whole gap between the two means) and 2.  Target chosen by seed 1865 among verbs with at least two other animals.  32-bit."""
from lib import *
import numpy as np
torch.set_num_threads(1)
exec(open("facts_common.py").read())
verbs = sorted({facts[a][0] for a in facts}); vb = {a: facts[a][0] for a in names}
with torch.no_grad():
    rows = {a: torch.stack([x[0, P[a]["site"]] for x in forward(ft, P[a]["ids"][:P[a]["site"] + 1])["resid"]]) for a in names}
rng = np.random.default_rng(SEED); R = {}
tg = {}
for a in names:
    cand = [v for v in verbs if v != vb[a] and sum(vb[b] == v for b in names if b != a) >= 2]; tg[a] = cand[rng.integers(len(cand))]
print("target verbs chosen, count by verb:", {v: sum(t == v for t in tg.values()) for v in verbs})
for alpha in (1.0, 2.0):
    print(f"alpha {alpha}:")
    for l in [0, 1, 2, 3, 5, 7, 9, 11]:
        to = 0; stay = 0
        for a in names:
            src = [b for b in names if b != a and vb[b] == vb[a]]; dst = [b for b in names if b != a and vb[b] == tg[a]]
            d = torch.stack([rows[b][l] for b in dst]).mean(0) - torch.stack([rows[b][l] for b in src]).mean(0) if src else torch.stack([rows[b][l] for b in dst]).mean(0) - rows[a][l]
            e = P[a]
            with torch.no_grad(): lg = forward(ft, e["ids"][:e["site"] + 1], patch=(l, e["site"], rows[a][l] + alpha * d))["logits"][0, -1]
            t = int(lg.argmax()); to += t == T.encode(" " + tg[a])[0]; stay += t == e["verb"]
        R[f"{alpha}_{l}"] = [to, stay]; print(f"   layer {l:2d}: becomes the target verb {to:2d} of 30, stays {stay:2d}", flush=True)
print("control: a random direction with the same length as d, a = 1 (seeded draws):")
g = torch.Generator().manual_seed(SEED)
for l in [0, 3, 7, 9, 11]:
    to = 0; stay = 0; chg = 0
    for a in names:
        src = [b for b in names if b != a and vb[b] == vb[a]]; dst = [b for b in names if b != a and vb[b] == tg[a]]
        d = torch.stack([rows[b][l] for b in dst]).mean(0) - torch.stack([rows[b][l] for b in src]).mean(0) if src else torch.stack([rows[b][l] for b in dst]).mean(0) - rows[a][l]
        r = torch.randn(d.shape, generator=g); r = r / r.norm() * d.norm(); e = P[a]
        with torch.no_grad(): lg = forward(ft, e["ids"][:e["site"] + 1], patch=(l, e["site"], rows[a][l] + r))["logits"][0, -1]
        t = int(lg.argmax()); to += t == T.encode(" " + tg[a])[0]; stay += t == e["verb"]
    R[f"random_{l}"] = [to, stay]; print(f"   layer {l:2d}: becomes the target verb {to:2d} of 30, stays the original {stay:2d}", flush=True)
save_json("steer", R)
