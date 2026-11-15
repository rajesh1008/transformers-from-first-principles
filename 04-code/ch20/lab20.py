"""Lab 20: Tiny's optimisers, schedules and four ways to break them, with a checkpoint.
Seed 1865 (Chapter 19's windows), 32-bit, two threads.  About a minute.  Stops with an error if a checkpoint fails."""
import _common
import hashlib, math
import numpy as np
import torch
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from tiny.train import train_torch
from tiny.optim import SGD, Adam, Muon, train_opt, newton_schulz, warmup_cosine, wsd, constant
from _common import novel_ids, tok, SEED
torch.set_num_threads(2)
nov = np.array(novel_ids(tok()))
rng = np.random.default_rng(SEED)
starts = rng.integers(0, len(nov) - 129, size=(400, 8))
batches = lambda s: torch.tensor(np.stack([nov[a:a + 129] for a in starts[s]]))
def fingerprint(model):
    h = hashlib.sha256()
    for k, v in model.tables().items():
        h.update(k.encode()); h.update(np.ascontiguousarray(v, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]
def fresh():
    return TorchTiny.from_numpy(Tiny(), dtype=torch.float32)
def first_step(opt):
    tt = fresh(); before = {k: v.detach().clone() for k, v in tt.t.items()}
    train_opt(tt, batches, 1, opt)
    d = torch.cat([(tt.t[k].detach() - before[k]).flatten() for k in tt.t])
    return tt, d
results = []
def check(name, ok):
    results.append(ok); print(f"  [{'pass' if ok else 'FAIL'}] {name}")

print("Part 1. one step of Tiny from its default start: the size of the change per number, divided by lr")
_, d = first_step(Adam(3e-3))
moved = d[d != 0].abs() / 3e-3
print(f"  Adam (b1 0.9, b2 0.95, corrected): {len(moved):,} numbers moved; |change|/lr median {moved.median():.4f}, min {moved.min():.4f}, max {moved.max():.4f}")
_, d_sgd = first_step(SGD(0.1))
r = d_sgd[d_sgd != 0].abs() / 0.1
print(f"  SGD lr 0.1: |change|/lr is the nudge itself: median {r.median():.2e}, max {r.max():.2e}")

print("Part 2. four breakages")
_, d1 = first_step(Adam(3e-3, b2=0.999, bias_correction=False))
b1r = float((d1[d1 != 0].abs() / 3e-3).median())
print(f"  1 no bias correction, b2 0.999: first step |change|/lr median {b1r:.4f} (0.1/sqrt(0.001) = {0.1 / math.sqrt(0.001):.4f})")
_, d2 = first_step(Adam(3e-3, eps=1e-2))
b2r = float((d2[d2 != 0].abs() / 3e-3).median())
print(f"  2 eps 1e-2 instead of 1e-8: first step |change|/lr median {b2r:.2e}; Adam has become slow gradient descent")
tt = fresh(); hist = train_opt(tt, batches, 60, Adam(3e-2), constant())
tw = fresh(); hw = train_opt(tw, batches, 60, Adam(3e-2), warmup_cosine(40, 400, 0.1))
print(f"  3 Adam lr 3e-2, 60 steps: no warmup batch loss at step 59 {hist[-1][1]:.4f} (largest {max(L for _, L in hist):.4f} at step {max(hist, key=lambda h: h[1])[0]});"
      f" with 40 warmup steps {hw[-1][1]:.4f} (largest {max(L for _, L in hw):.4f} at step {max(hw, key=lambda h: h[1])[0]})")
G = torch.randn(128, 512, generator=torch.Generator().manual_seed(SEED)) * 50
X = G.clone()
a, b, c = 3.4445, -4.7750, 2.0315
with torch.no_grad():
    for _ in range(5):
        A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
fin = bool(torch.isfinite(X).all())
sv_bad = float(torch.linalg.svdvals(X).max()) if fin else float("inf")
big = float(X.abs().max()) if fin else float("nan")
sv = torch.linalg.svdvals(newton_schulz(G))
print(f"  4 Newton-Schulz without dividing by the table's size first (a 128 x 512 table of spread 50): finite {fin}, largest singular value {sv_bad:.3g};"
      f" with it: {float(sv.min()):.3f} to {float(sv.max()):.3f}")

print("Part 3. checkpoints")
m = Tiny(); lt = float(m.loss(novel_ids(tok())[:1025]))
check(f"tiny/optim.py changes nothing about Tiny: {m.n_params:,} learned numbers, fingerprint {fingerprint(m)}, loss {lt:.4f}",
      m.n_params == 2766592 and fingerprint(m) == "19c8710677873d01" and round(lt, 4) == 7.6334)
torch.set_num_threads(1)                    # one thread: the only setting in which two runs are bit-for-bit equal here
ta, tb, tc = fresh(), fresh(), fresh()
train_torch(ta, batches, 10, 0.1); train_opt(tb, batches, 10, SGD(0.1))
diff = max(float((ta.t[k] - tb.t[k]).abs().max()) for k in ta.t)
torch.set_num_threads(2)
td, te = fresh(), fresh()
train_torch(td, batches, 10, 0.1); train_torch(te, batches, 10, 0.1)
diff2 = max(float((td.t[k] - te.t[k]).abs().max()) for k in td.t)
print(f"  (two runs of train_torch at two threads, same seed, 10 steps: largest difference {diff2:.3g})")
check(f"SGD(0.1) in train_opt equals Chapter 17's train_torch after 10 steps at one thread: largest difference {diff:g}", diff == 0.0)
check(f"Adam's first step moves every nudged number by lr: median |change|/lr {moved.median():.4f}",
      abs(float(moved.median()) - 1) < 1e-3)
check(f"without bias correction the first step is about 3.162 x lr (eps takes the last 0.1 per cent): {b1r:.4f}", abs(b1r - 0.1 / math.sqrt(0.001)) < 0.01)
f, w = warmup_cosine(40, 400, 0.1), wsd(40, 400, 0.2)
check(f"schedules: warmup+cosine {f(0):.3f} at step 0, {f(39):.3f} at 39, {f(399):.4f} at 399; WSD {w(319):.3f} at 319, {w(399):.4f} at 399",
      abs(f(0) - 0.025) < 1e-12 and f(39) == 1.0 and abs(f(399) - 0.1) < 1e-3 and w(319) == 1.0 and abs(w(399) - 0.0125) < 1e-9)
check(f"Newton-Schulz: singular values {float(sv.min()):.3f} to {float(sv.max()):.3f}, inside 0.6 to 1.3", 0.6 < float(sv.min()) and float(sv.max()) < 1.3)
n2 = sum(v.numel() for v in fresh().t.values())
check(f"Adam stores two numbers per learned number: {2 * n2:,}", 2 * n2 == 5_533_184)
print(f"\n{sum(results)} of {len(results)} checkpoints pass")
assert all(results)
