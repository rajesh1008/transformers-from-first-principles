"""Lab 21: Tiny's number formats, five ways to break 16-bit training, and a checkpoint.
Seed 1865 (Chapter 19's windows), one thread.  About a minute.  Stops with an error if a checkpoint fails."""
import _common
import hashlib, math
import numpy as np
import torch
torch.set_num_threads(1)
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from tiny.optim import Adam, constant
from tiny.precision import MixedTiny, LossScaler, train_mixed, round_bf16_stochastic, Adam16
from _common import novel_ids, tok, SEED
nov = np.array(novel_ids(tok()))
rng = np.random.default_rng(SEED)
starts = rng.integers(0, len(nov) - 129, size=(400, 8))
batches = lambda s: torch.tensor(np.stack([nov[a:a + 129] for a in starts[s]]))
def fingerprint(model):
    h = hashlib.sha256()
    for k, v in model.tables().items():
        h.update(k.encode()); h.update(np.ascontiguousarray(v, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]
results = []
def check(name, ok):
    results.append(ok); print(f"  [{'pass' if ok else 'FAIL'}] {name}")
bf = lambda x: float(torch.tensor(x, dtype=torch.float32).to(torch.bfloat16))
hf = lambda x: float(torch.tensor(x, dtype=torch.float32).to(torch.float16))

print("Part 1. predict, then read: six numbers in BF16 and FP16")
for x in [1 + 2 ** -8, 1 + 2 ** -7, 0.1, 7.6334, 65504.0, 70000.0]:
    print(f"  {x:<12.8g} BF16 {bf(x):<12.8g} FP16 {hf(x):<12.8g}")

print("\nPart 2. five breakages")
B = batches(0)
# 1. FP16 without loss scaling: the smallest nudges vanish
m = Tiny()
tt = MixedTiny.from_numpy(m, compute=torch.float16)
L = tt.loss(B); L.backward()
g16 = torch.cat([v.grad.flatten() for v in tt.t.values()])
tt32 = TorchTiny.from_numpy(m, dtype=torch.float32)
L32 = tt32.loss(B); L32.backward()
g32 = torch.cat([v.grad.flatten() for v in tt32.t.values()])
lost = int(((g16 == 0) & (g32 != 0)).sum())
tt = MixedTiny.from_numpy(m, compute=torch.float16)
L = tt.loss(B); (L * 2.0 ** 16).backward()
g16s = torch.cat([v.grad.flatten() for v in tt.t.values()]) / 2.0 ** 16
lost_s = int(((g16s == 0) & (g32 != 0)).sum())
rel = lambda a, b: float((a - b).norm() / b.norm())
print(f"  1. FP16 products, no loss scale: {lost:,} nudges that are not zero in 32-bit arrive as exactly 0; relative error {rel(g16, g32):.3g}")
print(f"     with the loss multiplied by 2^16 (and the nudges divided back): {lost_s:,}; relative error {rel(g16s, g32):.3g}")
# 2. a loss scale too large
tt = MixedTiny.from_numpy(m, compute=torch.float16)
sc = LossScaler(2.0 ** 24)
L = tt.loss(B); (L * sc.scale).backward()
grads = [v.grad for v in tt.t.values()]
bad = int(sum((~torch.isfinite(gg)).sum() for gg in grads))
ok = sc.unscale_ok(grads)
print(f"  2. loss scale 2^24: {bad:,} nudges are inf or nan; step taken: {ok}; scale now 2^{int(math.log2(sc.scale))}")
# 3. BF16 tables with no master copy: the gains never move
tt = TorchTiny.from_numpy(m, dtype=torch.bfloat16)
before = {k: v.detach().clone() for k, v in tt.t.items()}
train_mixed(tt, batches, 5, Adam(1e-3), constant(), mode="weights16")
gain_moved = sum(int((tt.t[k] != before[k]).sum()) for k in tt.t if k.endswith(".gain"))
tab_moved = sum(int((tt.t[k] != before[k]).sum()) for k in tt.t if tt.t[k].ndim == 2)
print(f"  3. BF16 tables, no master copy, Adam lr 1e-3 for 5 steps: gains changed {gain_moved} of 3,200; two-axis numbers changed {tab_moved:,}")
# 4. Adam's arithmetic in FP16
g = torch.tensor([1e-4, 3e-3], dtype=torch.float16)
v = (0.05 * g * g)
eps16 = torch.tensor(1e-8, dtype=torch.float16)
upd = (0.1 * g) / (v.sqrt() + eps16)
print(f"  4. Adam's first step in FP16 for nudges 1e-4 and 3e-3: v = {[float(x) for x in v]}, eps stored as {float(eps16)}, "
      f"m/(sqrt(v)+eps) = {[float(x) for x in upd]} (should be about 0.447 each without bias correction)")
# 5. the loss computed in BF16
with torch.autocast("cpu", dtype=torch.bfloat16):
    tta = TorchTiny.from_numpy(m, dtype=torch.float32)
    La = tta.loss(B)
Lm = MixedTiny.from_numpy(m).loss(B)
print(f"  5. the same batch's loss: 32-bit {float(L32.detach()):.6f}; mixed BF16 with a 32-bit softmax {float(Lm.detach()):.6f}; "
      f"CPU autocast {float(La.detach()):.6f} (dtype {La.dtype}); BF16 numbers near 7.6 are {2 ** -5} apart")
print("\nPart 3. checkpoints")
m = Tiny(); lt = float(m.loss(novel_ids(tok())[:1025]))
check(f"tiny/precision.py changes nothing about Tiny: {m.n_params:,} learned numbers, fingerprint {fingerprint(m)}, loss {lt:.4f}",
      m.n_params == 2766592 and fingerprint(m) == "19c8710677873d01" and round(lt, 4) == 7.6334)
mt = MixedTiny.from_numpy(m, compute=torch.float32)
d0 = abs(float(mt.loss(B)) - float(TorchTiny.from_numpy(m, dtype=torch.float32).loss(B)))
check(f"MixedTiny with 32-bit products is Tiny in 32-bit: loss difference {d0:.3g}", d0 < 1e-5)
def run10():
    tt = MixedTiny.from_numpy(Tiny()); h = train_mixed(tt, batches, 10, Adam(1e-3))
    return [L for _, L in h], torch.cat([v.detach().flatten() for v in tt.t.values()])
(a, wa), (b, wb) = run10(), run10()
check(f"mixed BF16 at one thread repeats bit for bit: 10 steps twice, largest table difference {float((wa - wb).abs().max())}",
      a == b and bool((wa == wb).all()))
d1 = abs(float(Lm.detach()) - float(L32.detach()))
check(f"mixed BF16's first loss is within 0.01 of 32-bit's: {float(Lm.detach()):.4f} against {float(L32.detach()):.4f}", d1 < 0.01)
gen = torch.Generator().manual_seed(SEED)
x = torch.full((100_000,), 1.001)
sr = float(round_bf16_stochastic(x, gen).float().mean())
check(f"stochastic rounding keeps the average: 100,000 roundings of 1.001 average {sr:.5f} (nearest gives {bf(1.001)})",
      abs(sr - 1.001) < 1e-4)
check(f"a loss scale of 2^16 brings back every nudge FP16 lost: {lost:,} zeros unscaled, {lost_s:,} scaled", lost > 0 and lost_s == 0)
st = Adam16(1e-3); tt = MixedTiny.from_numpy(Tiny()); train_mixed(tt, batches, 1, st)
nb = sum(t.numel() * t.element_size() for t in list(st.m.values()) + list(st.v.values()))
check(f"Adam with BF16 state stores 4 bytes per learned number: {nb:,} bytes", nb == 4 * 2766592)
print(f"\n{sum(results)} of {len(results)} checkpoints pass")
assert all(results)
