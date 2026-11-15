"""Section 7.4: how fast does one multiplication run as the matrices grow?  Square (n, n) @ (n, n),
64-bit numbers, all 8 threads and then 1 thread.  Build machine: CPU only, as of November 2026.
Run from 04-code/ch07:  python3 speed_by_size.py   (add --measure to time your own machine)"""
import _common
from _common import recorded
import os, sys, subprocess, json, time
import numpy as np
from tiny.linalg import intensity

SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
CODE = """
import time, json, numpy as np
out = {}
for n in %s:
    A = np.random.default_rng(1865).normal(size=(n, n)); B = A.T.copy()
    reps = max(3, int(2e8 // (2 * n ** 3)))
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps): A @ B
        best = min(best, (time.perf_counter() - t0) / reps)
    out[n] = 2 * n ** 3 / best / 1e9
print(json.dumps(out))
""" % SIZES

def run(threads):
    env = {**os.environ, "OPENBLAS_NUM_THREADS": str(threads)}
    r = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True, check=True)
    return json.loads(r.stdout)

def best_of(threads, tries=3):
    runs = [run(threads) for _ in range(tries)]          # the machine is shared: keep each size's best
    return {n: max(r[n] for r in runs) for n in runs[0]}
rec = recorded("speed_by_size", lambda: {"all": best_of(os.cpu_count()), "one": best_of(1), "threads": os.cpu_count()})
print(f"{'n':>6}{'operations':>16}{'numbers moved':>15}{'ops per number':>16}{'8 threads':>12}{'1 thread':>11}")
for n in SIZES:
    ops, moved = 2 * n ** 3, 3 * n * n
    print(f"{n:>6}{ops:>16,}{moved:>15,}{ops / moved:>16,.0f}{rec['all'][str(n)]:>12.1f}{rec['one'][str(n)]:>11.1f}")
print("(speeds in billions of operations per second; measured", rec["measured"] + ")")
print(f"best: {max(rec['all'].values()):.1f} with {rec['threads']} threads, {max(rec['one'].values()):.1f} with one")
