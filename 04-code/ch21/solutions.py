"""Chapter 21 exercises: by-hand answers checked in code (1 to 8), and the code exercises (9 to 12).
Exercise 9 and 11 read the .json files written by train_mixed.py (lr 1e-4 runs) and ex11_fp8.py; run those first."""
import _common
import json, os, math
import numpy as np
import torch
torch.set_num_threads(1)
from tiny.precision import round_bf16_stochastic, mx_round, round_fp4
from _common import HERE, SEED
J = lambda n: json.load(open(os.path.join(HERE, "outputs", n + ".json")))
print("1. six exponent bits, nine mantissa bits: gap above 1 = 2^-9 =", 2 ** -9, "; between 64 and 128:", 2 ** 6 * 2 ** -9,
      "; 100 + 0.05: half the gap is 0.0625 > 0.05, so the sum is 100")
bf = lambda x: float(torch.tensor(x, dtype=torch.float32).to(torch.bfloat16))
for w in (0.5, 0.005):
    up = 2.0 ** (math.floor(math.log2(w)) - 7); dn = up / 2 if w == 2 ** math.floor(math.log2(w)) else up
    print(f"2. BF16 at {w}: gap above {up:.3g}, below {dn:.3g}; a step changes it if larger than half the gap on its side "
          f"({dn / 2:.3g} downward, {up / 2:.3g} upward); check: {bf(w - dn / 2 * 1.01)} and {bf(w - dn / 2 * 0.99)}")
print("   a norm gain of 1.0: half the gap below is 2^-9 = 0.00195, so an Adam lr below about 0.002 (step about lr) never moves it")
lo = math.ceil(math.log2(6.1e-5 / 2e-6)); hi = math.floor(math.log2(65504 / 0.3))
print(f"3. typical 2e-6 into FP16's normal range needs S >= 2^{lo}; largest 0.3 stays below 65,504 for S <= 2^{hi}; "
      f"safe 2^{lo} to 2^{hi}; start a dynamic scaler at 2^{hi} and let it halve")
N = 70e9
for name, b in [("32-bit Adam", 16), ("mixed, master", 16), ("BF16 m, v", 12), ("8-bit state", 10), ("no master", 8)]:
    print(f"4. {name:14s} {b * N / 1e9:7,.0f} GB")
nd = 512
print(f"   over 512 devices: optimiser sharded {(4 + 12 / nd) * N / 1e9:.1f} GB, + nudges {(2 + 14 / nd) * N / 1e9:.1f} GB, + weights {16 * N / nd / 1e9:.2f} GB")
s, d, h, L = 8192, 8192, 64, 80
a1 = 17 * 2 * s * d * L; a2 = 2.5 * 2 * h * s * s * L; a3 = 2 * s * d * L
print(f"5. per sequence: {a1 / 1e9:.1f} GB per-token part + {a2 / 1e9:.1f} GB attention tables = {(a1 + a2) / 1e9:.1f} GB; "
      f"without tables {a1 / 1e9:.1f} GB; every layer recomputed {a3 / 1e9:.1f} GB")
print(f"6. MFU {1.2e15 / 4.0 / 1e15:.0%}; with recomputation, same useful work in 5.0 s: {1.2e15 / 5.0 / 1e15:.0%}")
lo7 = 1 - 2 ** -8; p = (1.0 - 0.999) / (1.0 - lo7)
gen = torch.Generator().manual_seed(SEED)
r = round_bf16_stochastic(torch.full((200_000,), 0.999), gen).float()
print(f"7. 0.999 lies between BF16 {lo7} and 1.0; P(down) = 0.001 / 2^-8 = {p:.3f}, P(1.0) = {1 - p:.3f}; expected "
      f"{1 - p * 2 ** -8:.6f}; 200,000 roundings (seed {SEED}): share down {float((r < 1).float().mean()):.3f}, mean {float(r.mean()):.6f}")
rng = np.random.default_rng(SEED)
blk = torch.tensor(rng.standard_normal(32), dtype=torch.float32); blk[5] = 40.0
out = mx_round(blk, 32, "FP4 E2M1")
keep = [i for i in range(32) if i != 5]
surv = [(round(float(blk[i]), 3), float(out[i])) for i in keep if out[i] != 0]
print(f"8. scale 2^(floor(log2 40) - 2) = 2^{math.floor(math.log2(40)) - 2} = {2 ** (math.floor(math.log2(40)) - 2)}; 40 / 8 = 5 is "
      f"a tie between 4 and 6 and is stored as {float(out[5])}; others survive only if |x| / 8 > 0.25, |x| > 2: "
      f"seed {SEED} block: {len(surv)} of 31 survive {surv}")
print()
try:
    a, b = J("train_mixed_fp32_lr1e-4"), J("train_mixed_w16_nearest_lr1e-4")
    print(f"9. lr 1e-4 constant: 32-bit {a['held_end']:.4f}, BF16 tables nearest {b['held_end']:.4f} (lost at the end {b['lost'][-1]:.3f}); "
          f"at 3e-5 the gap was 6.4299 vs 7.0999, at 1e-3 cosine 5.2684 vs 5.2832")
except FileNotFoundError:
    print("9. run train_mixed.py fp32 1e-4 const and w16_nearest 1e-4 const first")
v32 = torch.tensor(np.random.default_rng(SEED).uniform(0, 1, 4096), dtype=torch.float32)
exact = float(v32.double().sum()); v = v32.to(torch.bfloat16); exact16 = float(v.double().sum())
plain = torch.tensor(0.0, dtype=torch.bfloat16)
for x in v:
    plain = plain + x
s_ = torch.tensor(0.0, dtype=torch.bfloat16); c = torch.tensor(0.0, dtype=torch.bfloat16)
for x in v:
    y = x - c; t = s_ + y; c = (t - s_) - y; s_ = t
print(f"10. 4,096 numbers in [0, 1) stored in BF16 (seed {SEED}): exact sum {exact:.2f} ({exact16:.2f} after rounding the inputs to BF16); running sum in BF16 {float(plain):.2f}; "
      f"Kahan sum with both running numbers in BF16 {float(s_):.2f}")
for mo in ("tensor", "mx"):
    try:
        print(f"11. FP8 E4M3 products, {mo} scales: held-out {J('ex11_fp8_' + mo)['held_end']:.4f} (32-bit 5.2684)")
    except FileNotFoundError:
        print(f"11. run ex11_fp8.py {mo} first")
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
import tiny.torch_tiny as TT
def saved_bytes(tt, batch, layers):
    keep = tt.config["n_layers"]; tt.config["n_layers"] = layers; seen = {}
    def pack(x):
        if not (x.requires_grad and x.is_leaf):
            seen.setdefault((x.untyped_storage().data_ptr(), x.storage_offset(), tuple(x.shape), tuple(x.stride()), x.dtype), x.numel() * x.element_size())
        return x
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        L = tt.loss(batch)
    L.backward(); tt.config["n_layers"] = keep
    return sum(seen.values())
m = Tiny(); B = torch.tensor(np.random.default_rng(SEED).integers(0, 2048, size=(8, 129)))
def per_layer():
    a = saved_bytes(TorchTiny.from_numpy(m, dtype=torch.float32), B, layers=12)
    b2 = saved_bytes(TorchTiny.from_numpy(m, dtype=torch.float32), B, layers=6)
    return (a - b2) / 6
p0 = per_layer()
old = TT.ACT["gelu"]; TT.ACT["gelu"] = lambda x: torch.nn.functional.gelu(x, approximate="tanh")
p1 = per_layer(); TT.ACT["gelu"] = old
att = 2.01 * 8 * 4 * 128 * 128 * 4
print(f"12. one layer keeps {p0 / 1e6:.2f} MB with Tiny's GELU and {p1 / 1e6:.2f} MB with PyTorch's fused tanh GELU; "
      f"per token per width: {(p0 - att) / (4 * 8 * 128 * 128):.1f} and {(p1 - att) / (4 * 8 * 128 * 128):.1f} numbers")
