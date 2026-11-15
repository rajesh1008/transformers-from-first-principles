"""Chapter 21, sections 21.4 and 21.5: where Tiny's nudges and steps fall on the 16-bit number lines.
One forward and backward pass of Tiny's default start on the first batch of Chapter 19's window draw (seed 1865),
in 32-bit at one thread.  Then: which nudges FP16 loses, what a loss scale does, and which Adam steps a BF16
table cannot record."""
import _common
import numpy as np
import torch
torch.set_num_threads(1)
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from tiny.precision import fmt_info
from _common import novel_ids, tok, SEED, save_json
T = tok(); nov = np.array(novel_ids(T))
rng = np.random.default_rng(SEED)
starts = rng.integers(0, len(nov) - 129, size=(400, 8))
batch = torch.tensor(np.stack([nov[a:a + 129] for a in starts[0]]))
m = Tiny()
tt = TorchTiny.from_numpy(m, dtype=torch.float32)
L, G = tt.grads(batch)
g = np.concatenate([v.ravel() for v in G.values()])
nz = np.abs(g[g != 0])
print(f"Tiny's nudges on the first batch: {g.size:,} numbers, {int((g == 0).sum()):,} exactly zero (unreached position rows); loss {L:.4f}")
q = np.quantile(nz, [0.001, 0.01, 0.5, 0.99, 0.999])
print("  size of the non-zero nudges: 0.1% below {:.3g}, 1% below {:.3g}, median {:.3g}, 99% below {:.3g}, 99.9% below {:.3g}, largest {:.3g}".format(*q, nz.max()))
small = {}
for k, v in G.items():
    key = k.split(".")[-1] if k.startswith("h.") else k
    small[key] = small.get(key, 0) + int(((np.abs(v) < 6.1e-5) & (v != 0)).sum())
tot = sum(small.values())
print(f"  {tot:,} non-zero nudges are below 6.1e-5; by table: " +
      ", ".join(f"{k} {c / tot:.1%}" for k, c in sorted(small.items(), key=lambda kv: -kv[1])[:6]))
f16 = fmt_info("FP16")
print(f"\nFP16: smallest subnormal {f16['min_sub']:.3g}, smallest normal {f16['min_normal']:.3g}, largest {f16['max']:.0f}")
print(f"{'loss scale':>11s} {'become 0':>10s} {'subnormal':>10s} {'overflow':>10s}   (shares of the non-zero nudges, if stored in FP16)")
rows = []
for k in [0, 4, 8, 12, 16, 20, 24]:
    s = nz * 2.0 ** k
    st = torch.tensor(s, dtype=torch.float32).to(torch.float16).float().numpy()
    zero = float((st == 0).mean()); sub = float(((s < f16["min_normal"]) & (st != 0)).mean()); inf = float(np.isinf(st).mean())
    rows.append(dict(k=k, zero=zero, sub=sub, inf=inf))
    print(f"{'2^' + str(k):>11s} {zero:10.4%} {sub:10.4%} {inf:10.4%}")
print("  BF16 has FP32's exponent range: no nudge of Tiny's becomes 0 or overflows in BF16 "
      f"({int((torch.tensor(nz, dtype=torch.float32).to(torch.bfloat16) == 0).sum())} zeros)")
print("\nThe nudge on the loss itself is 1, and the nudge arriving at each next-token score is (share - 1 or share)/1,024:")
print(f"  1/1,024 = {1/1024:.3g}; a score whose share is 1e-4 gets {1e-4/1024:.3g}, which FP16 stores as {float(torch.tensor(1e-4/1024).half()):.3g}")

print("\nWhich steps a table stored in BF16 cannot record (w - step rounds back to w), by kind of table")
t = m.tables()
two = np.concatenate([v.ravel() for k, v in t.items() if v.ndim == 2])
gains = np.concatenate([v.ravel() for k, v in t.items() if k.endswith(".gain")])
biases = np.concatenate([v.ravel() for k, v in t.items() if v.ndim == 1 and not k.endswith(".gain")])
def lost(w, step):
    w16 = torch.tensor(w, dtype=torch.float32).to(torch.bfloat16)
    new = (w16.float() - step).to(torch.bfloat16)
    return float((new == w16).float().mean())
print(f"{'step size':>10s} {'two-axis tables':>16s} {'gains (1.0)':>12s} {'biases (0)':>11s}")
lrows = []
for step in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 2.5e-5, 1e-5]:
    a, b, c = lost(two, step), lost(gains, step), lost(biases, step)
    lrows.append(dict(step=step, tables=a, gains=b, biases=c))
    print(f"{step:10.1e} {a:16.1%} {b:12.1%} {c:11.1%}")
print(f"  ({two.size:,} numbers in two-axis tables, spread {np.sqrt((two ** 2).mean()):.4f}; {gains.size:,} gains; {biases.size:,} biases)")
print("  Adam's steps are about lr per number; the schedule in train_mixed.py starts at 0.025 x 1e-3 = 2.5e-5 and ends at 1e-4")
hist, edges = np.histogram(np.log2(nz), bins=80)
save_json("nudges", dict(hist=hist.tolist(), edges=edges.tolist(), scale=rows, lost=lrows, loss=L))
