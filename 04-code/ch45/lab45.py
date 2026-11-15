"""Lab 45: check the arithmetic of the chapter by simulation, in a second.  numpy, seed 1865.
1. A ring all-reduce simulated on 4 virtual ranks (reduce-scatter, then all-gather) gives every rank the exact sum, and
   each rank sends 2(W-1)/W of the buffer: the factor used in dist45.py and paper45.py.
2. A GPipe schedule simulated step by step (forward then backward, unit time per stage per micro-batch) has idle share
   (p-1)/(m+p-1) for p, m in {2,4,8} x {1,4,16}.
3. Splitting a feed-forward layer by columns then rows, with one sum at the end, equals the unsplit layer (W = 2, 4, 8).
4. Splitting a matrix product by rows of the input (data parallel) needs no communication for the forward pass.
5. ZeRO-3 memory per rank times N equals plain memory; ZeRO-1 and -2 lie between.
6. dist45.json: data-parallel first-step gradient within 1e-6 of one process; pipeline logits identical; tensor parallel
   within 1e-5; ranks agree exactly after 20 steps."""
import json, os, numpy as np
rng = np.random.default_rng(1865); ok = []
def ring(bufs):
    W = len(bufs); ch = [np.array_split(b.copy(), W) for b in bufs]; sent = 0
    for s in range(W - 1):                                   # reduce-scatter
        msgs = [(r, (r - s) % W, ch[r][(r - s) % W].copy()) for r in range(W)]
        for r, c, m in msgs: ch[(r + 1) % W][c] += m; sent += m.size
    for s in range(W - 1):                                   # all-gather
        msgs = [(r, (r + 1 - s) % W, ch[r][(r + 1 - s) % W].copy()) for r in range(W)]
        for r, c, m in msgs: ch[(r + 1) % W][c] = m; sent += m.size
    return [np.concatenate(c) for c in ch], sent / W
B = [rng.normal(size=1000) for _ in range(4)]; out, per = ring(B); tot = sum(B)
ok.append(all(np.allclose(o, tot) for o in out) and abs(per - 2 * 3 / 4 * 1000) < 1e-9)
def bubble(p, m):
    """Event simulation: forward of micro-batch j on stage s starts when stage s-1 has finished it and stage s is free;
    after the last forward on the last stage, backwards run from the last stage to the first, same rule.  Unit time each."""
    F = {}; free = [0] * p
    for j in range(m):
        for st in range(p):
            start = max(free[st], F.get((st - 1, j), 0)); F[(st, j)] = start + 1; free[st] = start + 1
    Bk = {}; t0 = max(F.values())
    for j in range(m):
        for st in reversed(range(p)):
            start = max(free[st], t0, Bk.get((st + 1, j), 0)); Bk[(st, j)] = start + 1; free[st] = start + 1
    T = max(Bk.values()); return (p * T - 2 * m * p) / (p * T)
ok.append(all(abs(bubble(p, m) - (p - 1) / (m + p - 1)) < 1e-12 for p in (2, 4, 8) for m in (1, 4, 16)))
x = rng.normal(size=(16, 32)); W1 = rng.normal(size=(32, 128)); W2 = rng.normal(size=(128, 32)); f = lambda z: np.maximum(z, 0)
full = f(x @ W1) @ W2
ok.append(all(np.allclose(sum(f(x @ W1[:, k * 128 // n:(k + 1) * 128 // n]) @ W2[k * 128 // n:(k + 1) * 128 // n] for k in range(n)), full) for n in (2, 4, 8)))
ok.append(np.allclose(np.vstack([x[k * 4:(k + 1) * 4] @ W1 for k in range(4)]), x @ W1))
P, N = 1e9, 64; z3 = 16 * P / N; z1 = (4 + 12 / N) * P; z2 = (2 + 14 / N) * P
ok.append(abs(z3 * N - 16 * P) < 1 and z3 < z2 < z1 < 16 * P)
d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "dist45.json")))
ok.append(d["dp_grad_maxdiff"] < 1e-6 and d["pp_maxdiff"] == 0.0 and d["tp_maxdiff"] < 1e-5 and d["dp_ranks_agree"] == 0.0)
for i, v in enumerate(ok, 1): print(f"check {i}: {'pass' if v else 'FAIL'}")
print(f"{sum(ok)} of {len(ok)} checks pass")
