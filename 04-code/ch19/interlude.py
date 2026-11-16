"""Chapter 19, Maths Interlude: adding n random numbers multiplies the spread by sqrt(n).
Part 1: coin flips of +1 or -1, summed n at a time, 100,000 sums for each n.
Part 2: one output of a layer, x . w, where x has 128 numbers of spread 1 and w has 128 numbers of spread s.
Part 3: dividing by sqrt(n) brings the spread back to that of a single number."""
import _common
import numpy as np
from _common import SEED, save_json, rms
rng = np.random.default_rng(SEED)
out = {"coins": {}, "dot": {}}
print("Part 1: sums of n coin flips (+1 or -1), 100,000 sums each")
print("   n   spread of the sum   sqrt(n)   spread after dividing by sqrt(n)")
for n in (1, 2, 4, 16, 64, 128, 512, 2048):
    flips = rng.choice([-1.0, 1.0], size=(100_000, n))
    S = flips.sum(axis=1)
    out["coins"][n] = [float(S.std()), float(np.sqrt(n)), float((S / np.sqrt(n)).std())]
    print(f"{n:5d}   {S.std():17.3f}   {np.sqrt(n):7.3f}   {(S / np.sqrt(n)).std():8.3f}")
print("\nPart 2: one output x . w, x of 128 numbers with spread 1, w of 128 numbers with spread s (20,000 pairs)")
x = rng.normal(0, 1, size=(20_000, 128))
for s in (0.02, 1 / np.sqrt(128), np.sqrt(2 / 128), 0.2, 1.0):
    w = rng.normal(0, s, size=(20_000, 128))
    y = (x * w).sum(axis=1)
    out["dot"][f"{s:.5f}"] = [float(y.std()), float(s * np.sqrt(128))]
    print(f"  s = {s:.4f}: spread of the output {y.std():.4f}; the rule s x sqrt(128) says {s * np.sqrt(128):.4f}")
print("\nPart 3: the check-yourself numbers")
for n, s in ((512, 0.02), (512, 1 / np.sqrt(512)), (128, 0.02)):
    print(f"  fan-in {n}, spread {s:.4f}: output spread {s * np.sqrt(n):.4f}")
save_json("interlude", out)
