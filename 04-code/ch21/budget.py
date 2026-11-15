"""Chapter 21, sections 21.9 and 21.10: where the time of a training step goes, and what the arithmetic costs.
One thread, fastest of five, on the book's build machine (dated).  Part 1 times Tiny's step (forward, backward,
Adam) in 32-bit, mixed BF16, pure BF16 and with each layer recomputed.  Part 2 counts the step's floating-point
operations from the table shapes.  Part 3 measures the machine's best rate on one large product in 32-bit and
BF16, and Part 4 divides.  Part 5 is arithmetic for a 7-billion-number run (no measurement)."""
import _common
import time, datetime, json
import numpy as np
import torch
torch.set_num_threads(1)
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from tiny.optim import Adam
from tiny.precision import MixedTiny
from memory_budget import Ckpt
from _common import SEED, save_json
print(f"{datetime.date.today()}; one thread; fastest of five; batch 8 windows x 128 predictions")
gen = np.random.default_rng(SEED)
B = torch.tensor(gen.integers(0, 2048, size=(8, 129)))
m = Tiny()
def timed(model, opt, k=5):
    tt = model.tt if isinstance(model, Ckpt) else model
    F, Bk, O = [], [], []
    for _ in range(k + 1):
        for v in tt.t.values():
            v.grad = None
        t0 = time.perf_counter(); L = model.loss(B); t1 = time.perf_counter()
        L.backward(); t2 = time.perf_counter()
        opt.step(tt.t) if not isinstance(opt, str) else None; t3 = time.perf_counter()
        F.append(t1 - t0); Bk.append(t2 - t1); O.append(t3 - t2)
    return min(F[1:]) * 1e3, min(Bk[1:]) * 1e3, min(O[1:]) * 1e3
rows = {}
print(f"{'recipe':44s} {'forward ms':>10s} {'backward ms':>11s} {'Adam ms':>8s} {'step ms':>8s}")
for name, mk in [("32-bit", lambda: TorchTiny.from_numpy(m, dtype=torch.float32)),
                 ("mixed BF16 (products 16-bit)", lambda: MixedTiny.from_numpy(m)),
                 ("pure BF16 (no Adam step timed: see text)", lambda: TorchTiny.from_numpy(m, dtype=torch.bfloat16)),
                 ("32-bit, each layer recomputed", lambda: Ckpt(TorchTiny.from_numpy(m, dtype=torch.float32))),
                 ("mixed BF16, each layer recomputed", lambda: Ckpt(MixedTiny.from_numpy(m)))]:
    model = mk()
    opt = Adam(1e-3) if "pure" not in name else "none"
    f, b, o = timed(model, opt)
    rows[name] = (f, b, o)
    print(f"{name:44s} {f:10.1f} {b:11.1f} {o:8.1f} {f + b + o:8.1f}")
# Part 2: operations
b, n, d, h, V, Lr = 8, 128, 128, 4, 2048, 12
per_tok_layer = 2 * (4 * d * d + 2 * d * 4 * d)            # W_q, W_k, W_v, W_o and the two feed-forward tables
att = 2 * 2 * n * d                                           # scores and blend, every (seat, seat) pair computed
head = 2 * d * V
fwd = b * n * (Lr * (per_tok_layer + att) + head)
print(f"\nPart 2. operations: forward {fwd / 1e9:.2f} GFLOP (layers' tables {b * n * Lr * per_tok_layer / 1e9:.2f}, "
      f"attention scores and blends {b * n * Lr * att / 1e9:.2f}, output scores {b * n * head / 1e9:.2f}); "
      f"a step is about 3 x forward = {3 * fwd / 1e9:.2f} GFLOP")
print(f"  the 6ND rule with N = 2,766,592 and D = 1,024 tokens: {6 * 2766592 * 1024 / 1e9:.2f} GFLOP "
      f"(it counts the position table as arithmetic and leaves out attention's n^2 work)")
# Part 3: best rate
peak = {}
for dt, nm in [(torch.float32, "32-bit"), (torch.bfloat16, "BF16")]:
    X = torch.randn(1024, 1024).to(dt); Y = torch.randn(1024, 1024).to(dt)
    ts = []
    for _ in range(6):
        t0 = time.perf_counter(); X @ Y; ts.append(time.perf_counter() - t0)
    peak[nm] = 2 * 1024 ** 3 / min(ts[1:]) / 1e9
    print(f"Part 3. one 1,024 x 1,024 product in {nm}: {peak[nm]:.1f} GFLOP/s")
print("\nPart 4. Tiny's share of the machine's best rate (step operations / step time / best rate)")
for name, pk in [("32-bit", "32-bit"), ("mixed BF16 (products 16-bit)", "BF16"), ("32-bit, each layer recomputed", "32-bit"),
                 ("mixed BF16, each layer recomputed", "BF16")]:
    f, bb, o = rows[name]
    useful = 3 * fwd
    rate = useful / ((f + bb + o) / 1e3) / 1e9
    print(f"  {name:36s} {rate:6.1f} GFLOP/s of useful work = {rate / peak[pk]:.0%} of the {pk} best")
print("\nPart 5. a 7-billion-number model on 2 trillion tokens (arithmetic only; the sustained rate is an assumption)")
N7, D7 = 6.74e9, 2e12
for nm, k in [("6ND", 6), ("with every layer recomputed, 8ND", 8)]:
    F = k * N7 * D7
    print(f"  {nm:34s} {F:.3g} FLOP; at an assumed 4e14 FLOP/s sustained per accelerator: {F / 4e14 / 3600:,.0f} accelerator-hours")
save_json("budget", dict(date=str(datetime.date.today()), rows=rows, fwd=fwd, peak=peak))
