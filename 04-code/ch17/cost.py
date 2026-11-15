"""Chapter 17, section 17.8: what the backward pass costs, and why nobody wiggles 2,766,592 numbers.

1. One table product Y = X W and its two backward products, timed (one thread, fastest of five).
2. Tiny's forward pass and forward-plus-backward pass over 1,024 tokens in PyTorch, 32-bit and 64-bit.
3. The wiggle alternative: NumPy Tiny's loss over 128 predictions, timed, times 2 x 2,766,592.
Seed 1865 for the random ids and tables.  The book's build machine, one thread, 10 November 2026."""
import _common
import time, datetime
import numpy as np
import torch
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from _common import best_of, save_json, novel_ids, SEED
torch.set_num_threads(1)
rng = np.random.default_rng(SEED)
print(f"seed {SEED}; torch threads {torch.get_num_threads()}; {datetime.date.today()}")

n, d, k = 1024, 128, 512
X = torch.tensor(rng.normal(size=(n, d)), dtype=torch.float32)
Wt = torch.tensor(rng.normal(size=(d, k)), dtype=torch.float32)
dY = torch.tensor(rng.normal(size=(n, k)), dtype=torch.float32)
f = best_of(lambda: X @ Wt, 9)
bx = best_of(lambda: dY @ Wt.T, 9)
bw = best_of(lambda: X.T @ dY, 9)
fl = 2 * n * d * k
print(f"\none table product, X (1,024 x 128) times W (128 x 512): {fl:,} FLOPs forward")
print(f"  forward  Y = X W           {f:.3f} ms")
print(f"  backward nudge on X = dY W^T {bx:.3f} ms ({fl:,} FLOPs)")
print(f"  backward nudge on W = X^T dY {bw:.3f} ms ({fl:,} FLOPs)")
print(f"  backward / forward: FLOPs 2.00, time {(bx + bw) / f:.2f}")

m = Tiny()
ids = novel_ids()[:1025]
res = {}
for name, dt in (("32-bit", torch.float32), ("64-bit", torch.float64)):
    tt = TorchTiny.from_numpy(m, dtype=dt)
    with torch.no_grad():
        tf = best_of(lambda: tt.loss(ids), 5)
    def fb():
        for v in tt.t.values():
            v.grad = None
        tt.loss(ids).backward()
    tfb = best_of(fb, 5)
    res[name] = dict(forward_ms=tf, fwd_bwd_ms=tfb, ratio=(tfb - tf) / tf)
    print(f"\nTiny in PyTorch, {name}, 1,024 tokens: forward {tf:.1f} ms; forward + backward {tfb:.1f} ms;"
          f" backward alone {tfb - tf:.1f} ms = {(tfb - tf) / tf:.2f} x the forward")
N = m.n_params
print(f"\nthe 6N rule: forward about 2N = {2*N:,} FLOPs per token, backward about 4N = {4*N:,}; training about 6N = {6*N:,}")
print(f"  one pass through the novel ({len(novel_ids()):,} tokens): 6 x {N:,} x {len(novel_ids()):,} = {6*N*len(novel_ids()):.3e} FLOPs")

ids129 = ids[:129]
tw = best_of(lambda: m.loss(ids129), 5)
total = 2 * N * tw / 1000
print(f"\nwiggling instead: NumPy Tiny's loss over 128 predictions takes {tw:.1f} ms; two per learned number:"
      f" 2 x {N:,} x {tw:.1f} ms = {total:,.0f} s = {total/86400:.1f} days for one set of nudges")
tt = TorchTiny.from_numpy(m, dtype=torch.float64)
def fb128():
    for v in tt.t.values():
        v.grad = None
    tt.loss(ids129).backward()
ta = best_of(fb128, 5)
print(f"autograd over the same 128 predictions, 64-bit: {ta:.1f} ms for all {N:,} nudges, {total*1000/ta:,.0f} times faster")
save_json("cost", dict(table=dict(f=f, bx=bx, bw=bw), tiny=res, wiggle_ms=tw, wiggle_days=total/86400, autograd128_ms=ta,
                       date=str(datetime.date.today())))
