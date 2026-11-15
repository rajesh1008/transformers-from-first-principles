"""Chapter 25, sections 25.6 and 25.7: comparing two checkpoints honestly.
Every model is scored on the same predictions (all 436 Looking-Glass windows, 55,808 predictions; all 31 Carroll held-out
windows, 3,968), so differences can be paired.  For each pair: the difference in loss; an unpaired 95 per cent interval
(each model's window bootstrap taken separately, standard errors added in quadrature); the paired interval (whole windows
resampled together, 2,000 times, seed 1865); and a sign test over windows (how many windows favour each model).
Then the training noise the paired interval cannot see: the same comparison (AdamW 0.1 against none) under three reading
orders.  32-bit, one thread.  Writes outputs/compare.json."""
import _common
import os, math, json
import numpy as np, torch
torch.set_num_threads(1)
from tiny import eval as E
from _pool import corpora
from _common import HERE, save_json
c = corpora()
TX = dict(glass=E.windows(c["glass"]), carroll=E.windows(c["held_b"]))
CK24 = os.path.join(HERE, "..", "ch24", "outputs", "ckpt"); CK25 = os.path.join(HERE, "outputs", "ckpt")
F = {"1e-3 2,500": (CK24, "wsd_1e-3_2000_b2000"), "1e-3 1,250": (CK24, "wsd_1e-3_2000_b1000"), "2e-3 2,500": (CK24, "wsd_2e-3_2000_b2000"),
     "wd0.1 2,500": (CK25, "wsd_1e-3_2000_wd0.1_b2000"), "o7 2,500": (CK25, "wsd_1e-3_2000_o7_b2000"), "wd0.1 o7 2,500": (CK25, "wsd_1e-3_2000_wd0.1_o7_b2000"),
     "o11 2,500": (CK25, "wsd_1e-3_2000_o11_b2000"), "wd0.1 o11 2,500": (CK25, "wsd_1e-3_2000_wd0.1_o11_b2000"),
     "wd0.1 1,250": (CK25, "wsd_1e-3_2000_wd0.1_b1000"), "o7 1,250": (CK25, "wsd_1e-3_2000_o7_b1000"), "wd0.1 o7 1,250": (CK25, "wsd_1e-3_2000_wd0.1_o7_b1000"),
     "o11 1,250": (CK25, "wsd_1e-3_2000_o11_b1000"), "wd0.1 o11 1,250": (CK25, "wsd_1e-3_2000_wd0.1_o11_b1000")}
S = {}
for k, (d, f) in F.items():
    tt = E.load(os.path.join(d, f + ".pt"))
    S[k] = {t: E.score(tt, W) for t, W in TX.items()}
print("losses on every window (Looking-Glass 436, Carroll held-out 31):")
for k in F:
    print(f"  {k:16} Looking-Glass {S[k]['glass']['surprise'].mean():.4f}   Carroll {S[k]['carroll']['surprise'].mean():.4f}")
out = dict(loss={k: {t: float(S[k][t]["surprise"].mean()) for t in TX} for k in F}, pairs={})

def compare(a, b, t):
    x, y = S[a][t]["surprise"], S[b][t]["surprise"]; w = S[a][t]["window"]
    ba, bb = E.bootstrap(x, w), E.bootstrap(y, w)
    d = x.mean() - y.mean(); se_u = math.hypot(ba["se"], bb["se"])
    p = E.paired(x, y, w)
    wd = np.bincount(w, weights=x - y)
    st = E.sign_test(wd)
    r = dict(diff=float(d), unpaired=(float(d - 1.96 * se_u), float(d + 1.96 * se_u)), paired=(p["low"], p["high"]), se_paired=p["se"], se_unpaired=se_u,
             windows=st["n"], a_better=int((wd < 0).sum()), p_sign=st["p"])
    print(f"  {a} minus {b} on {t}: {d:+.4f}; unpaired 95% {r['unpaired'][0]:+.4f} to {r['unpaired'][1]:+.4f}; paired {p['low']:+.4f} to {p['high']:+.4f} "
          f"(se {se_u:.4f} -> {p['se']:.4f}); {r['a_better']} of {st['n']} windows favour the first, sign test p = {st['p']:.2g}")
    return r
print("\npaired comparisons:")
for a, b in (("1e-3 2,500", "1e-3 1,250"), ("1e-3 2,500", "2e-3 2,500"), ("wd0.1 2,500", "1e-3 2,500"), ("o7 2,500", "1e-3 2,500"),
             ("wd0.1 o7 2,500", "o7 2,500"), ("wd0.1 o11 2,500", "o11 2,500"), ("o11 2,500", "1e-3 2,500")):
    for t in ("glass", "carroll"):
        out["pairs"][f"{a} | {b} | {t}"] = compare(a, b, t)
print("\nthe weight-decay effect under three reading orders (Looking-Glass, 2,500 and 1,250 steps):")
for L_ in ("2,500", "1,250"):
    no = [f"1e-3 {L_}", f"o7 {L_}", f"o11 {L_}"]; wd = [f"wd0.1 {L_}", f"wd0.1 o7 {L_}", f"wd0.1 o11 {L_}"]
    for t in ("glass", "carroll"):
        dn = [S[a][t]["surprise"].mean() for a in no]; dw = [S[a][t]["surprise"].mean() for a in wd]
        diffs = np.array(dw) - np.array(dn)
        print(f"  {L_} steps, {t:8}: no decay {', '.join(f'{x:.4f}' for x in dn)} (sd {np.std(dn, ddof=1):.4f}); "
              f"decay {', '.join(f'{x:.4f}' for x in dw)} (sd {np.std(dw, ddof=1):.4f}); differences {', '.join(f'{x:+.4f}' for x in diffs)}, "
              f"mean {diffs.mean():+.4f}, sd {diffs.std(ddof=1):.4f}")
        out[f"wd_orders_{L_}_{t}"] = dict(no=dn, wd=dw, diffs=diffs.tolist())
save_json("compare", out)
