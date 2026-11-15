"""Chapter 37, 37.7: abstention as a decision with a price.  No model is run: this reads outputs/name_share.json and
outputs/agree2.json (SmolLM2-135M, the 30 real countries).  Three doubt signals: the mean share of the greedy sentence,
the share on the name token, and the agreement of ten samples at T=0.7.  For each signal and each threshold, answer when
the signal is at or above it and decline otherwise; report coverage (answered of 30), right among answered, and the score
under a grading that gives +1 for right, 0 for declined and -k for wrong (k = 0, 1, 3).  The greedy answer is the one
given; agreement decides only whether to give it."""
from _common import *
import numpy as np
ns = {r["country"]: r for r in load_json("name_share")["SmolLM2-135M"] if r["group"] != "invented"}
ag = {r["country"]: r for r in load_json("agree2")["SmolLM2-135M T=0.7"] if r["group"] != "invented"}
C = list(ns); right = np.array([ns[c]["right"] for c in C])
S = {"sentence share": np.array([ns[c]["mean_share"] for c in C]),
     "name-token share": np.array([ns[c]["name_share"] or 0.0 for c in C]),
     "agreement of ten at T=0.7": np.array([ag[c]["agree"] for c in C])}
print(f"greedy answers right: {int(right.sum())} of 30", flush=True)
R = {}
for name, v in S.items():
    print(f"{name}:", flush=True); R[name] = []
    for th in sorted(set(np.round(v, 3))) [::1]:
        ans = v >= th; n = int(ans.sum()); r = int((right & ans).sum()); w = n - r
        row = dict(th=float(th), answered=n, right=r, wrong=w, score0=r, score1=r - w, score3=r - 3 * w); R[name].append(row)
    best = {k: max(R[name], key=lambda x: x[k]) for k in ("score1", "score3")}
    for th in (0.0, 0.2, 0.3, 0.4, 0.5):
        ans = v >= th; n = int(ans.sum()); r = int((right & ans).sum()); w = n - r
        print(f"   threshold {th:.1f}: answer {n:2d} of 30, right {r:2d}, wrong {w:2d}; score at k=0 {r:3d}, k=1 {r - w:3d}, k=3 {r - 3 * w:3d}", flush=True)
    for k, b in best.items():
        print(f"   best threshold for {k} (k={k[-1]}): {b['th']:.3f} -> answer {b['answered']}, right {b['right']}, wrong {b['wrong']}, score {b[k]}", flush=True)
from itertools import combinations
for name, v in S.items():
    # AUROC: chance that a random right answer has a higher signal than a random wrong one (ties count half)
    pos = v[right]; neg = v[~right]
    au = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
    print(f"AUROC of {name} for separating right from wrong greedy answers: {au:.2f}", flush=True); R[name + " auroc"] = float(au)
save_json("abstain", R)
