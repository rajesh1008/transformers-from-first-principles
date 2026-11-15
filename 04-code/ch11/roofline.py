"""Chapter 11, section 6: the shape of this machine.  Measures, on the book's CPU, (a) matrix-multiply
throughput, (b) main-memory copy bandwidth, (c) how many exponentials per second NumPy computes, and
(d) the time of each step of plain attention at n = 2,048, so the arithmetic-to-bytes balance of this
machine can be compared with attention's.  Fastest of five, recorded; --measure to re-time.
Run from 04-code/ch11:  python3 roofline.py"""
from _common import *
import time

def best(f, k=5):
    t0 = 1e9
    for _ in range(k):
        t = time.perf_counter(); f(); t0 = min(t0, time.perf_counter() - t)
    return t0

def measure():
    rng = np.random.default_rng(SEED)
    A = rng.normal(size=(2048, 2048)).astype(np.float32); Bm = rng.normal(size=(2048, 2048)).astype(np.float32)
    t_mm = best(lambda: A @ Bm)
    big = rng.normal(size=64 * 2**20 // 4).astype(np.float32)          # 64 MiB
    out = np.empty_like(big)
    t_cp = best(lambda: np.copyto(out, big))
    t_exp = best(lambda: np.exp(big, out=out))
    t_add = best(lambda: np.add(big, 1.0, out=out))
    n, d = 2048, 64
    Q, K, V = (rng.normal(size=(n, d)).astype(np.float32) for _ in range(3))
    S = Q @ K.T
    t_s = best(lambda: Q @ K.T)
    t_max = best(lambda: S.max(1, keepdims=True))
    E = np.exp(S - S.max(1, keepdims=True))
    t_e = best(lambda: np.exp(S - S.max(1, keepdims=True)))
    t_div = best(lambda: E / E.sum(1, keepdims=True))
    P = E / E.sum(1, keepdims=True)
    t_pv = best(lambda: P @ V)
    return {"matmul_gflops": round(2 * 2048**3 / t_mm / 1e9, 1),
            "copy_gbps": round(2 * big.nbytes / t_cp / 1e9, 1),
            "exp_per_s_billion": round(big.size / t_exp / 1e9, 2),
            "add_per_s_billion": round(big.size / t_add / 1e9, 2),
            "attn_ms": {"scores QK^T": round(t_s * 1e3, 2), "row max": round(t_max * 1e3, 2),
                        "exp(S - max)": round(t_e * 1e3, 2), "sum and divide": round(t_div * 1e3, 2),
                        "blend PV": round(t_pv * 1e3, 2)}}
r = recorded("roofline", measure)
print("book build machine, float32, NumPy 2.5.4, measured", r["measured"])
print(f"  matrix multiply 2048 x 2048: {r['matmul_gflops']} GFLOP/s")
print(f"  memory copy 64 MiB (read + write counted): {r['copy_gbps']} GB/s")
print(f"  exponentials: {r['exp_per_s_billion']} billion per second | additions: {r['add_per_s_billion']} billion per second")
bal = r['matmul_gflops'] / r['copy_gbps']
print(f"  machine balance: {bal:.1f} FLOPs per byte (below this, a step is limited by memory)")
print(f"  matmul FLOPs per exponential: {r['matmul_gflops'] / r['exp_per_s_billion']:.0f}")
print("\nplain attention, one head, n = 2,048, d = 64, float32, step by step (ms):")
tot = sum(r["attn_ms"].values())
for k, v in r["attn_ms"].items():
    print(f"  {k:<16} {v:>7.2f}  ({v / tot:5.1%})")
print(f"  total            {tot:>7.2f}")
n, d = 2048, 64
print(f"  the two multiplies do {4 * n * n * d / 1e6:.0f} MFLOP; the softmax steps do about {5 * n * n / 1e6:.0f} M simple operations and {n * n / 1e6:.1f} M exponentials")
