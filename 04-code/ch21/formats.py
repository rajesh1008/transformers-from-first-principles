"""Chapter 21, sections 21.1 to 21.3: number formats by hand and on Tiny's tables.  No training, no timing."""
import _common
import math
import numpy as np
import torch
from tiny.model import Tiny
from tiny.precision import FORMATS, fmt_info, round_to, scaled_round, mx_round
from _common import SEED

names = list(FORMATS)
print("Part 1. the formats: 1 sign bit + exponent bits + mantissa bits")
print(f"{'format':10s} {'bits':>4s} {'exp':>4s} {'man':>4s} {'largest':>11s} {'smallest normal':>16s} {'smallest':>11s} {'gap above 1':>12s}")
for n in names:
    f = fmt_info(n)
    print(f"{n:10s} {f['bits']:4d} {f['exp']:4d} {f['man']:4d} {f['max']:11.4g} {f['min_normal']:16.4g} {f['min_sub']:11.4g} {f['eps']:12.4g}")

print("\nPart 2. five numbers stored in each format (nearest; FP8 E4M3 saturates at 448, FP4 at 6)")
xs = [0.1, 1 / 3, 1000.7, 70000.0, 3e-6]
print(f"{'format':10s} " + " ".join(f"{x:>14.7g}" for x in xs))
for n in names:
    r = round_to(torch.tensor(xs, dtype=torch.float32), n)
    print(f"{n:10s} " + " ".join(f"{float(v):14.7g}" for v in r))

def gap(x, man):
    return 2.0 ** (math.floor(math.log2(x)) - man)

print("\nPart 3. the gap between neighbouring numbers near three sizes Tiny meets")
print(f"{'format':10s} {'near 1 (gains)':>15s} {'near 0.02 (tables)':>19s} {'near 4e-4 (nudges)':>19s}")
for n in ["FP32", "BF16", "FP16", "FP8 E4M3"]:
    man = FORMATS[n][1]
    print(f"{n:10s} {gap(1.0, man):15.3g} {gap(0.02, man):19.3g} {gap(4e-4, man):19.3g}")

print("\nPart 4. adding 0.001 to 1.0, 1,000 times (the answer is 2.0)")
for n, dt in [("FP32", torch.float32), ("BF16", torch.bfloat16), ("FP16", torch.float16)]:
    s = torch.tensor(1.0, dtype=dt)
    for _ in range(1000):
        s = s + torch.tensor(0.001, dtype=dt)
    print(f"  running sum kept in {n}: {float(s):.6f}")
rng = np.random.default_rng(SEED)
v = torch.tensor(rng.uniform(0, 1, 4096), dtype=torch.float32)
exact = float(v.double().sum())
for n, dt in [("BF16", torch.bfloat16), ("FP16", torch.float16)]:
    s = torch.tensor(0.0, dtype=dt)
    for x in v.to(dt):
        s = s + x
    s32 = float(v.to(dt).float().sum())
    print(f"  4,096 numbers between 0 and 1 (seed {SEED}): exact {exact:.2f}; sum kept in {n} {float(s):.2f}; "
          f"stored in {n}, summed in FP32 {s32:.2f}")

one = torch.ones(1, 4096, dtype=torch.bfloat16)
print(f"  the same 4,096 numbers as one BF16 product (ones @ v): {float(one @ v.to(torch.bfloat16)[:, None]):.2f} "
      f"(the product sums in 32-bit and rounds once, to BF16)")

print("\nPart 5. Tiny's tables stored in each format: relative error (rms of the error / rms of the table)")
m = Tiny(); t = m.tables()
W = torch.tensor(t["h.0.ff.W_1"], dtype=torch.float32)
E = torch.tensor(t["E"], dtype=torch.float32)
g = torch.tensor(t["h.0.ln_1.gain"], dtype=torch.float32)
def rel(a, b):
    return float(((a - b) ** 2).mean().sqrt() / (b ** 2).mean().sqrt())
print(f"{'format':28s} {'W_1 (layer 0)':>14s} {'E':>10s}")
for n in ["BF16", "FP16", "FP8 E4M3", "FP8 E5M2", "FP4 E2M1"]:
    print(f"{n + ', no scale':28s} {rel(round_to(W, n), W):14.3g} {rel(round_to(E, n), E):10.3g}")
for n in ["FP8 E4M3", "FP4 E2M1"]:
    print(f"{n + ', one scale per table':28s} {rel(scaled_round(W, n), W):14.3g} {rel(scaled_round(E, n), E):10.3g}")
    print(f"{n + ', MX blocks of 32':28s} {rel(mx_round(W, 32, n), W):14.3g} {rel(mx_round(E, 32, n), E):10.3g}")

print("\nPart 6. rows of spread 1 (as after a norm) stored in 8 and 4 bits, with and without one feature 100 times larger;")
print("        relative error of the other 127 features, and the share of them stored as exactly 0")
ids = rng.integers(0, 2048, size=64)
X = E[torch.tensor(ids)].clone()
X = X / X.pow(2).mean(-1, keepdim=True).sqrt()
Xo = X.clone(); Xo[:, 7] *= 100.0
keep = torch.ones(X.shape[1], dtype=torch.bool); keep[7] = False
print(f"{'how the rows are stored':34s} {'plain rows':>16s} {'with one large feature':>26s}")
for desc, f in [("BF16", lambda a: round_to(a, "BF16")),
                ("FP8 E4M3, one scale per table", lambda a: scaled_round(a, "FP8 E4M3")),
                ("FP8 E4M3, MX blocks of 32", lambda a: mx_round(a, 32, "FP8 E4M3")),
                ("FP4 E2M1, one scale per table", lambda a: scaled_round(a, "FP4 E2M1")),
                ("FP4 E2M1, MX blocks of 32", lambda a: mx_round(a, 32, "FP4 E2M1"))]:
    a, b = f(X)[:, keep], f(Xo)[:, keep]
    z1, z2 = float((a == 0).float().mean()), float((b == 0).float().mean())
    print(f"{desc:34s} {rel(a, X[:, keep]):9.3g} ({z1:4.0%} 0) {rel(b, Xo[:, keep]):17.3g} ({z2:4.0%} 0)")
