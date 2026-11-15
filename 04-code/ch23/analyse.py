"""Chapter 23: read every outputs/train_*.json and print the chapter's tables and fits (outputs/analyse.txt, .json).
Held-out loss is the Carroll held-out set (Chapter 22's 16 windows of the last 4,000 tokens of Sylvie and Bruno
Concluded) unless a line says Looking-Glass.  N is the non-embedding count unless a line says total."""
import _common
import glob, json, os
import numpy as np
from tiny import scaling as S
from sizes import N, FIVE, counts, TOKENS_PER_STEP, WIDTHS
from _common import HERE, save_json
R = {}
for f in glob.glob(os.path.join(HERE, "outputs", "train_*.json")):
    r = json.load(open(f)); R[r["tag"]] = r
def get(size, steps, seed=1865, lr=None):
    t = f"{size}_{steps}" + (f"_s{seed}" if seed != 1865 else "") + (f"_lr{lr}" if lr else "")
    return R.get(t)
OUT = {}
SEEDS = [1865, 7, 11]
print("Part 1. the six sizes (four layers, head width 16, context 128)")
for s in WIDTHS:
    c = counts(s)
    print(f"  {s} width {c['width']:3d}  non-embedding {c['non_embedding']:>9,d}  total {c['total']:>9,d}")
def sweep(lr, label):
    """Parts 2 to 5 for one peak learning rate (lr None = 1e-3)."""
    res = {}
    print(f"\nPart 2{label}. one pass of the pool (384 steps, 393,216 tokens), three reading orders, peak lr {lr or '1e-3'}")
    print("  size        N     C=6ND     Carroll: 1865    7       11     mean   spread | Looking-Glass mean spread | sentence (1865)")
    rows = {}
    for s in FIVE + ["s6"]:
        rs = [get(s, 384, sd, lr) for sd in SEEDS]
        if any(r is None for r in rs):
            print(f"  {s} incomplete"); continue
        cb = [r["carroll"] for r in rs]; gl = [r["glass"] for r in rs]
        rows[s] = dict(N=N[s], C=rs[0]["C"], carroll=cb, glass=gl, mean=float(np.mean(cb)), spread=float(np.ptp(cb)),
                       gmean=float(np.mean(gl)), gspread=float(np.ptp(gl)), sec=[r["seconds"] for r in rs],
                       sentence=rs[0].get("sentence"), curve=rs[0]["curve"])
        sent = f"{rs[0]['sentence']:.4f}" if rs[0].get("sentence") is not None else "-"
        print(f"  {s} {N[s]:>9,d} {rs[0]['C']:.2e}   " + "  ".join(f"{v:.4f}" for v in cb) +
              f"  {np.mean(cb):.4f}  {np.ptp(cb):.4f} | {np.mean(gl):.4f} {np.ptp(gl):.4f} | {sent}")
    res["rows"] = rows
    five = [s for s in FIVE if s in rows]
    if len(five) < 5:
        return res
    x = np.array([N[s] for s in five], float); y = np.array([rows[s]["mean"] for s in five])
    print(f"\nPart 3{label}. fits to the five means (Carroll held-out)")
    ln = S.fit_line(x, y)
    print(f"  straight line on log-log axes  L = {ln['A']:.3f} N^-{ln['alpha']:.4f}   (each doubling x{S.doublings(ln['alpha']):.4f})")
    fl = S.fit_floor(x, y)
    print(f"  with a floor                   L = {fl['E']:.3f} + {fl['A']:.2f} N^-{fl['alpha']:.4f}   squared misses {fl['sse']:.2e}"
          f"{'  (floor at the edge of the search)' if fl['at_edge'] else ''}{'  (floor at zero)' if fl['at_zero'] else ''}")
    for s in five:
        print(f"    {s}: measured {rows[s]['mean']:.4f}  line {S.predict(ln, N[s]):.4f}  floor fit {S.predict(fl, N[s]):.4f}")
    res["fit_line"], res["fit_floor"] = ln, fl
    print("  how the fit moves with the number of points (smallest sizes first)")
    res["by_points"] = {}
    for k in (3, 4, 5):
        f = S.fit_floor(x[:k], y[:k]); g = S.fit_line(x[:k], y[:k])
        res["by_points"][k] = dict(floor=f, line=g)
        print(f"    {k} points: floor fit E {f['E']:.3f}  A {f['A']:.2f}  alpha {f['alpha']:.4f}  | line alpha {g['alpha']:.4f}"
              f"{'  (floor at edge)' if f['at_edge'] else ''}")
    print("  each reading order's five numbers fitted alone")
    res["by_order"] = {}
    for i, sd in enumerate(SEEDS):
        yi = np.array([rows[s]["carroll"][i] for s in five])
        f = S.fit_floor(x, yi); g = S.fit_line(x, yi)
        res["by_order"][sd] = dict(floor=f, line=g)
        print(f"    order {sd:4d}: floor fit E {f['E']:.3f}  A {f['A']:.2f}  alpha {f['alpha']:.4f}  | line alpha {g['alpha']:.4f}")
    print("  leave one out (floor fit on four means, miss on the fifth)")
    res["loo"] = []
    for d in S.leave_one_out(x, y):
        res["loo"].append(d)
        print(f"    without {five[d['left_out']]}: alpha {d['fit']['alpha']:.4f}  E {d['fit']['E']:.3f}  miss on it {d['miss']:+.4f}")
    if "s6" in rows:
        print(f"\nPart 4{label}. predicting the sixth size (width 192, N 1,776,768), held back from every fit")
        for name, f in (("line", ln), ("floor", fl)):
            pr = float(S.predict(f, N["s6"]))
            print(f"  {name:5s} fit predicts {pr:.4f}; measured {rows['s6']['mean']:.4f} (orders {', '.join(f'{v:.4f}' for v in rows['s6']['carroll'])}, spread {rows['s6']['spread']:.4f}); miss {rows['s6']['mean'] - pr:+.4f}")
            res[f"s6_pred_{name}"] = pr
        x6 = np.append(x, N["s6"]); y6 = np.append(y, rows["s6"]["mean"])
        f6 = S.fit_floor(x6, y6); res["fit_floor_six"] = f6; res["fit_line_six"] = S.fit_line(x6, y6)
        print(f"  floor fit with all six: E {f6['E']:.3f}  A {f6['A']:.2f}  alpha {f6['alpha']:.4f}; line alpha {res['fit_line_six']['alpha']:.4f}")
    print("  extrapolations far past the data (no run exists to check them)")
    for Nx in (2_373_376, 10_000_000, 100_000_000):
        print(f"    N {Nx:>11,d}: line {S.predict(ln, Nx):.3f}   floor {S.predict(fl, Nx):.3f}")
    print(f"\nPart 5{label}. the same five means with embeddings counted (total learned numbers)")
    xt = np.array([counts(s)["total"] for s in five], float)
    ft = S.fit_floor(xt, y); lt = S.fit_line(xt, y)
    res["fit_total"] = dict(floor=ft, line=lt)
    print(f"  line alpha {lt['alpha']:.4f} (non-embedding {ln['alpha']:.4f});  floor fit E {ft['E']:.3f} A {ft['A']:.2f} alpha {ft['alpha']:.4f}")
    return res
OUT["lr1e-3"] = sweep(None, "a")
OUT["lr3e-3"] = sweep("3e-3", "b")
rp = {}
for base in ("s1_384", "s3_384", "s5_384"):
    runs = [R.get(base)] + [R.get(base + sfx) for sfx in ("_r2", "_r3")]
    if all(runs):
        rp[base] = [r["carroll"] for r in runs]
if rp:
    print("\n  the same configuration run three times (order 1865, lr 1e-3; nothing changed between runs)")
    for t, v in rp.items():
        print(f"    {t}: " + "  ".join(f"{x:.4f}" for x in v) + f"   spread {max(v) - min(v):.4f}")
    OUT["repro"] = rp
print("\nPart 4c. s6 again: other peak rates at one pass, and longer runs (order 1865)")
OUT["s6_extra"] = {}
for lr in ("3e-4", "5e-4", None, "3e-3"):
    r = get("s6", 384, 1865, lr)
    if r:
        OUT["s6_extra"][f"lr_{lr or '1e-3'}"] = r["carroll"]
        print(f"  s6 384 steps, peak lr {lr or '1e-3'}: Carroll {r['carroll']:.4f}")
for st in (773, 1546):
    r6, r5 = get("s6", st), get("s5", st)
    if r6:
        OUT["s6_extra"][f"steps_{st}"] = r6["carroll"]
        cmp_ = f";  s5 at the same steps {r5['carroll']:.4f}" if r5 else ""
        print(f"  s6 {st} steps ({r6['passes']:.2f} passes, lr 1e-3): Carroll {r6['carroll']:.4f}{cmp_}")
print("\nPart 6. three compute budgets, each spent five ways (isoFLOP), held-out loss at the end of each run")
OUT["isoflop"] = {}
for base, lr in ((192, None), (768, None), (3072, None), (768, "3e-3"), (3072, "3e-3")):
    pts = []
    for s in FIVE:
        st = round(base * N["s3"] / N[s]); r = get(s, st, 1865, lr)
        if r is None:
            continue
        pts.append((s, N[s], st, r["D"], r["C"], r["passes"], r["carroll"], r["glass"]))
    if not pts:
        continue
    C = np.mean([p[4] for p in pts])
    print(f"  budget C = {C:.2e} operations (6ND), peak lr {lr or '1e-3'}")
    for s, n, st, D, c, ps, cb, gl in pts:
        print(f"    {s} N {n:>8,d}  steps {st:>6,d}  D {D:>10,d}  tokens per number {D / n:7.1f}  passes {ps:5.2f}  Carroll {cb:.4f}  Looking-Glass {gl:.4f}")
    m = S.isoflop_minimum([p[1] for p in pts], [p[6] for p in pts]) if len(pts) >= 3 else None
    best = min(pts, key=lambda p: p[6])
    key = f"{base}_{lr or '1e-3'}"
    OUT["isoflop"][key] = dict(C=C, points=pts, parabola=m, best=best[0])
    if m and m["N_opt"]:
        Dopt = C / (6 * m["N_opt"])
        print(f"    lowest measured: {best[0]}; parabola bottom N {m['N_opt']:,.0f} (inside the sizes: {m['inside']}), "
              f"L {m['L_opt']:.4f}, D = C/6N {Dopt:,.0f}, tokens per number {Dopt / m['N_opt']:.1f}")
        OUT["isoflop"][key]["D_opt"] = Dopt
    elif m:
        print("    the parabola opens downwards: no bottom")
print("\nPart 7. one size (s3, N 199,040) read for longer: tokens against held-out loss")
OUT["data_axis"] = []
for st in (96, 192, 384, 768, 1536, 3072):
    r = get("s3", st)
    if r is None:
        continue
    OUT["data_axis"].append((st, r["D"], r["passes"], r["carroll"], r["glass"], r["curve"]))
    print(f"  steps {st:5d}  D {r['D']:>9,d}  passes {r['passes']:5.2f}  Carroll {r['carroll']:.4f}  Looking-Glass {r['glass']:.4f}  "
          f"last-24 batch {np.mean(r['train'][-24:]):.4f}")
da = OUT["data_axis"]
if len(da) >= 3:
    one = [d for d in da if d[2] <= 1.01]
    if len(one) >= 3:
        f = S.fit_line([d[1] for d in one], [d[3] for d in one])
        print(f"  straight line through the one-pass points: L = {f['A']:.3f} D^-{f['alpha']:.4f}")
        for d in da:
            print(f"    D {d[1]:>9,d}: line {S.predict(f, d[1]):.4f}  measured {d[3]:.4f}  measured - line {d[3] - S.predict(f, d[1]):+.4f}")
        OUT["data_line"] = f
for lr in (None, "3e-3"):
  r1 = get("s1", 12140, 1865, lr)
  if r1:
    cv = r1["curve"]
    print(f"\n  s1 at peak lr {lr or '1e-3'} for 12,140 steps ({r1['passes']:.1f} passes): held-out Carroll along the run " +
          ", ".join(f"{(c[0] + 1) * 8 / 3074:.1f}p {c[2]:.3f}" for c in cv[1:]))
print("\nPart 8. learning rate: three sizes at five peak rates (384 steps, order 1865)")
OUT["lr"] = {}
for s in ("s1", "s3", "s5"):
    vals = []
    for lr in ("3e-4", None, "3e-3", "6e-3", "1e-2"):
        r = get(s, 384, 1865, lr)
        vals.append(r["carroll"] if r else None)
    OUT["lr"][s] = vals
    print(f"  {s}: " + "  ".join(f"{l} {v:.4f}" if v is not None else f"{l} -" for l, v in zip(("3e-4", "1e-3", "3e-3", "6e-3", "1e-2"), vals)))
print("\n  too few steps: s4 (N 446,016) stopped early, each with its own warmup and decay")
OUT["short"] = {}
for st in (48, 96, 343, 384):
    r = get("s4", st)
    if r:
        OUT["short"][st] = r["carroll"]
        print(f"    s4 {st:4d} steps: Carroll {r['carroll']:.4f}")
print("\nPart 9. what 6ND leaves out: time per step against 6 N x 1,024 (one thread, 384-step runs, order 1865)")
OUT["time"] = {}
for s in FIVE + ["s6"]:
    r = get(s, 384)
    if not r:
        continue
    d = WIDTHS[s]
    out_ops = 6 * d * 2048 * TOKENS_PER_STEP
    att_ops = 3 * 4 * (4 * 64 * d) * TOKENS_PER_STEP   # per layer and token, scores 2 x 64 x d and blend 2 x 64 x d (64 = average seats read under the causal mask); backward twice the forward; four layers
    six = 6 * N[s] * TOKENS_PER_STEP
    OUT["time"][s] = dict(ms=r["ms_per_step"], six=six, out=out_ops, att=att_ops)
    print(f"  {s}: {r['ms_per_step']:6.1f} ms/step; 6N x 1,024 = {six / 1e9:6.3f} G; output table adds {out_ops / 1e9:6.3f} G "
          f"({out_ops / six:.2f} of 6N); attention scores and blend add {att_ops / 1e9:.3f} G; "
          f"G per second counting 6N only {six / r['ms_per_step'] / 1e6:5.2f}, counting all {(six + out_ops + att_ops) / r['ms_per_step'] / 1e6:5.2f}")
save_json("analyse", OUT)
