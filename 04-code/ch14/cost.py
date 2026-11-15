"""Chapter 14, section 14.7: what depth costs.  (a) Time for one forward pass of Tiny over the first 1,024 tokens
of the novel, at 1 to 24 layers (fastest of 5).  (b) Memory: the learned numbers at 8 and 4 bytes, and the peak
memory NumPy allocates during one forward pass (tracemalloc), with the shares tables counted by hand.
(c) Depth against width at the same learned-number budget for the layers (about 2.37 million):
48 layers of 64, 12 of 128, 3 of 256, heads of 32 numbers each; time for 1,024 tokens at once and for
one token at a time (the generating case, previewing Chapter 30), and the number of steps that must
run one after another.  Untrained throughout: this measures cost only, never quality.
Timings are recorded once (outputs/cost_timing.json); --measure re-times on your machine.
The book ran it with one thread (OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1), because with all eight cores
the shared build machine gave timings that moved by up to a factor of two between runs.
Run from 04-code/ch14:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 cost.py"""
from _common import *
import tracemalloc
from tiny.model import Tiny

t = tok()
ids = novel_ids(1024, t)

def measure():
    """Every configuration is timed once per round, five rounds, interleaved, and the fastest round kept,
    so that a slow spell on the machine affects every configuration alike."""
    import time
    def once(f):
        t0 = time.perf_counter(); f(); return (time.perf_counter() - t0) * 1000
    models = {f"d{L}": Tiny(n_layers=L) for L in [1, 2, 4, 8, 12, 24]}
    for L, d in [(48, 64), (3, 256)]:
        models[f"s{L}x{d}"] = Tiny(n_layers=L, d_model=d, n_heads=d // 32)
    models["s12x128"] = models["d12"]
    jobs = {}
    for k, mm in models.items():
        jobs[k + ":1024"] = (lambda mm=mm: mm.forward(ids))
        if k.startswith("s"):
            jobs[k + ":one"] = (lambda mm=mm: [mm.forward(ids[:1]) for _ in range(20)])
    m = models["d12"]
    X = m.blocks[0](m.embed(ids))
    jobs["embed"] = lambda: m.embed(ids)
    jobs["output"] = lambda: m.scores_from(m.top_rows(X))
    best = {k: float("inf") for k in jobs}
    for _ in range(5):
        for k, f in jobs.items():
            best[k] = min(best[k], once(f))
    out = {"depth": {}, "shape": {}, "embed_ms": best["embed"], "output_ms": best["output"]}
    for L in [1, 2, 4, 8, 12, 24]:
        out["depth"][str(L)] = {"ms": best[f"d{L}:1024"], "n_params": models[f"d{L}"].n_params}
    for L, d in [(48, 64), (12, 128), (3, 256)]:
        mm = models[f"s{L}x{d}"]
        out["shape"][f"{L}x{d}"] = {"layers_params": sum(b.n_params for b in mm.blocks), "total": mm.n_params,
                                    "ms_1024": best[f"s{L}x{d}:1024"], "ms_one_token": best[f"s{L}x{d}:one"] / 20}
    return out

r = recorded("cost_timing", measure)
print(f"seed {SEED}; first 1,024 tokens of the novel; fastest of 5 interleaved rounds, measured {r['measured']} on the book's build machine (CPU, one thread)")
print("(a) one forward pass against depth")
print("  layers | learned numbers |     ms | ms per layer | tokens per second")
base = r["depth"]["1"]["ms"]
for L, v in r["depth"].items():
    print(f"  {int(L):>6} | {v['n_params']:>15,} | {v['ms']:6.1f} | {v['ms'] / int(L):12.1f} | {1024 / v['ms'] * 1000:>17,.0f}")
print(f"  token and position lookup {r['embed_ms']:.2f} ms; ln_f and the 2,048 scores per seat {r['output_ms']:.1f} ms")

print("\n(b) memory")
m = Tiny()
n = m.n_params
print(f"  learned numbers {n:,}: {n * 8 / 1e6:.1f} MB at 8 bytes (NumPy's default), {n * 4 / 1e6:.1f} MB at 4 bytes (float32), {n * 2 / 1e6:.1f} MB at 2")
for L in [1, 12]:
    mm = Tiny(n_layers=L)
    tracemalloc.start(); mm.forward(ids); cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    print(f"  peak memory allocated during one forward pass, {L:>2} layer(s): {peak / 1e6:6.1f} MB")
sh = 4 * 1024 * 1024 * 8
print(f"  by hand: one layer's four share tables of 1,024 x 1,024 at 8 bytes = {sh / 1e6:.1f} MB; kept for all 12 layers (as training must) = {12 * sh / 1e6:.0f} MB")
rows = 1024 * (128 * 6 + 512 * 2) * 8
print(f"  by hand: one layer's other intermediate tables (six of 1,024 x 128, two of 1,024 x 512) = {rows / 1e6:.1f} MB; twelve layers = {12 * rows / 1e6:.0f} MB")
print(f"  the 1,024 x 2,048 scores at the top: {1024 * 2048 * 8 / 1e6:.1f} MB")

print("\n(c) depth against width, the layers' learned numbers held near 2.37 million (heads of 32)")
print("  shape (layers x width) | layers' numbers | whole model | ms, 1,024 tokens | ms, one token | steps in sequence")
for k, v in r["shape"].items():
    L = int(k.split("x")[0])
    print(f"  {k:>22} | {v['layers_params']:>15,} | {v['total']:>11,} | {v['ms_1024']:16.1f} | {v['ms_one_token']:13.2f} | {2 * L:>17}")
