"""Chapter 11, section 5: the three shelves of the book's CPU.  Reads the cache sizes the operating system
reports (lscpu) and times reading arrays of growing size (np.dot of an array with itself), so the speed of each shelf shows in the result
(fastest of 5 per size, recorded once; --measure to re-time).  float32.  Run:  python3 shelves.py"""
from _common import *
import time, subprocess, os
info = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
caches = {k: v.strip() for k, v in (line.split(":", 1) for line in info.splitlines() if "cache" in line and ":" in line)}
print("cache sizes reported by lscpu on the book's build machine:")
for k in ("L1d cache", "L2 cache", "L3 cache"):
    print(f"  {k}: {caches.get(k, 'not reported')}")
sizes_kib = [16, 32, 256, 1024, 4096, 32768, 262144, 1048576]
def measure():
    res = {}
    for kib in sizes_kib:
        a = np.ones(kib * 1024 // 4, dtype=np.float32)
        reps = max(1, (64 * 2**20) // a.nbytes)
        best = 1e9
        for _ in range(5):
            t = time.perf_counter()
            for _ in range(reps):
                np.dot(a, a)
            best = min(best, (time.perf_counter() - t) / reps)
        res[str(kib)] = round(a.nbytes / best / 1e9, 1)
    return res
r = recorded("shelves", measure)
print(f"\nreading an array once (np.dot of the array with itself, float32), GB/s, fastest of 5, measured {r['measured']}:")
for kib in sizes_kib:
    label = f"{kib} KiB" if kib < 1024 else f"{kib // 1024} MiB"
    print(f"  {label:>9}: {r[str(kib)]:>6} GB/s")
