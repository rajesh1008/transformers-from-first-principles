"""Where This Bites: serving rotary positions with settings the model was not trained with.
No error is raised in any case; only the agreements change.  Run: python3 bites.py"""
import _common
import numpy as np
from tiny.embed import apply_rotary, clock_speeds, dot

def bf16(x):                       # round float32 numbers to bfloat16 (8 significant bits), as many GPUs do
    b = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    b = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return b.astype(np.uint32).view(np.float32)

if __name__ == "__main__":
    d = 128
    rng = np.random.default_rng(1865)
    Q = rng.normal(size=(300, d)); K = rng.normal(size=(300, d))
    def score(q, k, gap, speeds, layout="pairs", sign=1):
        if layout == "halves":                    # pair number i with number i + d/2, not with its neighbour
            order = [j for i in range(d // 2) for j in (i, i + d // 2)]
            q, k = q[order], k[order]
        return dot(apply_rotary([q], [0], speeds=speeds)[0], apply_rotary([k], [sign * gap], speeds=speeds)[0])
    trained = clock_speeds(d, 10000.0)
    cases = {"wrong base (500,000)": dict(speeds=clock_speeds(d, 500000.0)),
             "pairs laid out as halves": dict(speeds=trained, layout="halves"),
             "clocks turning backwards": dict(speeds=trained, sign=-1)}
    print("average size of an agreement (trained settings):",
          round(float(np.mean([abs(score(q, k, 10, trained)) for q, k in zip(Q, K)])), 2))
    for name, kw in cases.items():
        row = []
        for gap in [0, 1, 10, 100, 1000]:
            diff = np.mean([abs(score(q, k, gap, **kw) - score(q, k, gap, trained)) for q, k in zip(Q, K)])
            row.append(f"gap {gap}: {diff:5.2f}")
        print(f"{name:>25}: average change  " + "   ".join(row))

    for pos in [100, 257, 1000, 1001, 1023]:
        print(f"position {pos:>4} stored as float16 -> {float(np.float16(pos)):6.0f}, as bfloat16 -> {float(bf16(pos)):6.0f}")
