"""Lab 23: predict three numbers, break a scaling-law fit five ways, and run a checkpoint.
Reads the training runs in outputs/ (run run_all.sh, run_after.sh and analyse.py first).  One thread.  About 20 s."""
import _common
import json, os, glob, subprocess, sys
import numpy as np
from tiny import scaling as S
from sizes import N, FIVE, counts, WIDTHS
from _common import HERE
R = {json.load(open(f))["tag"]: json.load(open(f)) for f in glob.glob(os.path.join(HERE, "outputs", "train_*.json"))}
SEEDS = [1865, 7, 11]
def tag(s, st, sd=1865):
    return f"{s}_{st}" + (f"_s{sd}" if sd != 1865 else "")
mean = {s: np.mean([R[tag(s, 384, sd)]["carroll"] for sd in SEEDS]) for s in FIVE + ["s6"]}
x = np.array([N[s] for s in FIVE], float); y = np.array([mean[s] for s in FIVE])
base = S.fit_floor(x, y); line = S.fit_line(x, y)
print("Part 1. predict, then read")
print(f"  doubling N from s3 to about s4 under the line fit multiplies the loss by 2^-alpha = {S.doublings(line['alpha']):.4f}")
print(f"  6ND for s5 at one pass: {S.compute(N['s5'], 384 * 1024):.4e} operations")
print(f"  the output table's share of s1's arithmetic beside 6N: 2 d V / 2 N = {WIDTHS['s1'] * 2048 / N['s1']:.2f}")
print("\nPart 2. five breakages")
print("  Breakage 1: one size trained for too few steps (s4, each run with its own warmup and decay)")
for st in (48, 96, 384):
    r = R[f"s4_{st}"]
    print(f"    s4 at {st:3d} steps: held-out {r['carroll']:.4f}; the line through the five means puts s4 at {S.predict(line, N['s4']):.4f}")
y1 = y.copy(); y1[3] = R["s4_96"]["carroll"]
f1 = S.fit_floor(x, y1); l1 = S.fit_line(x, y1)
print(f"    with s4 at 96 steps in the fit: line alpha {line['alpha']:.4f} -> {l1['alpha']:.4f};  floor fit alpha {base['alpha']:.4f} -> {f1['alpha']:.4f}")
print(f"    predicted s6: {S.predict(base, N['s6']):.4f} -> {S.predict(f1, N['s6']):.4f}  (measured mean {mean['s6']:.4f})")
print("  Breakage 2: a compute count that leaves out the output table")
for s in FIVE:
    d = WIDTHS[s]; extra = 6 * d * 2048
    print(f"    {s}: 6N per token {6 * N[s]:>10,d}; output table adds {extra:>9,d} ({extra / (6 * N[s]):.2f} x)")
r_small = (6 * N["s1"] + 6 * 32 * 2048) / (6 * N["s1"]); r_big = (6 * N["s5"] + 6 * 128 * 2048) / (6 * N["s5"])
print(f"    an 'equal-compute' pair s1 and s5 under 6ND really differs by {r_small / r_big:.2f} x in arithmetic")
print("  Breakage 3: extrapolating past the data")
for Nx in (1e7, 1e8, 1e9):
    print(f"    N {Nx:.0e}: line {S.predict(line, Nx):.3f}, floor fit {S.predict(base, Nx):.3f}")
print("  Breakage 4: embeddings counted")
xt = np.array([counts(s)["total"] for s in FIVE], float)
lt = S.fit_line(xt, y); ft = S.fit_floor(xt, y)
print(f"    line alpha {line['alpha']:.4f} (non-embedding) -> {lt['alpha']:.4f} (total); floor fit {base['alpha']:.4f} -> {ft['alpha']:.4f}")
print("  Breakage 5: reading a long run's middle as if it were a short run")
long = R["s3_3072"]; short = R["s3_384"]
mid = [c for c in long["curve"] if c[0] == 383][0]
print(f"    s3 at step 384 inside the 3,072-step cosine run: {mid[2]:.4f};  s3 trained for 384 steps with its own decay: {short['carroll']:.4f}")
print("\nPart 3. checkpoints")
ok = []
def check(name, cond, detail):
    ok.append(bool(cond)); print(f"  [{'pass' if cond else 'FAIL'}] {name}: {detail}")
from tiny.model import Tiny
m = Tiny(); p = m.count()
check("tiny/scaling.py changes nothing about Tiny", sum(p.values()) == 2_766_592, f"{sum(p.values()):,d} learned numbers")
check("the six sizes are counted from their tables", [counts(s)["non_embedding"] for s in WIDTHS] == [50368, 112416, 199040, 446016, 791296, 1776768],
      ", ".join(f"{counts(s)['non_embedding']:,d}" for s in WIDTHS))
xs = np.array([1e4, 2e4, 4e4, 8e4, 1.6e5]); planted = S.fit_floor(xs, 3.0 + 9.0 * xs ** -0.35)
check("fit_floor recovers a planted law (synthetic: E 3, A 9, alpha 0.35)", abs(planted["alpha"] - 0.35) < 2e-3 and abs(planted["E"] - 3) < 2e-3,
      f"E {planted['E']:.4f} alpha {planted['alpha']:.4f}")
pb = S.isoflop_minimum([1e4, 3e4, 1e5, 3e5], [0.1 * (np.log(n) - np.log(5e4)) ** 2 + 4 for n in [1e4, 3e4, 1e5, 3e5]])
check("isoflop_minimum finds a planted bottom at N = 50,000 (synthetic)", abs(pb["N_opt"] - 5e4) < 1, f"{pb['N_opt']:,.1f}")
miss = np.abs(y - S.predict(base, x)).max(); spread = min(np.ptp([R[tag(s, 384, sd)]["carroll"] for sd in SEEDS]) for s in FIVE)
check("every size sits on the floor fit within the smallest order spread", miss < spread, f"largest miss {miss:.4f} < {spread:.4f}")
pr = subprocess.run([sys.executable, "repro_probe.py"], capture_output=True, text=True, cwd=HERE).stdout
fp32 = [l for l in pr.splitlines() if l.startswith("fp32")][0]
check("32-bit products repeat exactly at one thread (BF16 products do not; see repro_probe.py)", "step None" in fp32, fp32)
check("6ND for s5 at one pass", abs(S.compute(N["s5"], 393216) - 1.8669e12) < 1e8, f"{S.compute(N['s5'], 393216):.4e}")
print(f"\n  the fitted exponent (five means, lr 1e-3): floor fit {base['alpha']:.2f}, straight line {line['alpha']:.2f}")
print(f"\n{sum(ok)} of {len(ok)} checkpoints pass")
if not all(ok):
    raise SystemExit(1)
