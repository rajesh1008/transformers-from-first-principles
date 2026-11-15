"""Chapter 12, section 3: the activations, side by side.  ReLU, GELU (GPT-2's tanh form and the exact
form), SiLU, and the gated forms built from them.  Run from 04-code/ch12:  python3 activations.py"""
from _common import *
from tiny.ffn import relu, gelu, gelu_exact, silu

xs = np.array([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3], dtype=float)
print("value at x:        " + "".join(f"{x:>8.1f}" for x in xs))
for name, f in [("ReLU", relu), ("GELU (tanh form)", gelu), ("GELU (exact)", gelu_exact), ("SiLU", silu)]:
    print(f"  {name:<17}" + "".join(f"{v:>8.3f}" for v in f(xs)))

fine = np.linspace(-6, 6, 120001)
for name, f in [("GELU (exact)", gelu_exact), ("SiLU", silu)]:
    y = f(fine); i = y.argmin()
    print(f"{name}: lowest value {y[i]:.4f} at x = {fine[i]:.3f}; negative for every x below 0")
print(f"GELU tanh form against exact form, largest difference on [-6, 6]: {np.abs(gelu(fine) - gelu_exact(fine)).max():.1e}")
d = 1e-6
print("slopes by central difference (ReLU has no slope at 0; the two sides average 0.5):")
for name, f in [("ReLU", relu), ("GELU (exact)", gelu_exact), ("SiLU", silu)]:
    s = [(f(np.array([x + d]))[0] - f(np.array([x - d]))[0]) / (2 * d) for x in (-2.0, 0.0, 2.0)]
    print(f"slope of {name:<13} at -2, 0, 2: {[round(float(v), 3) for v in s]}")

rng = np.random.default_rng(SEED)
z = rng.normal(size=1_000_000)
for name, f in [("ReLU", relu), ("GELU (exact)", gelu_exact), ("SiLU", silu)]:
    y = f(z) if name != "GELU (exact)" else f(z[:200_000])
    print(f"on standard normal inputs, {name:<13}: share of outputs above zero {np.mean(y > 0):.3f}, "
          f"share below zero {np.mean(y < 0):.3f}, mean output {y.mean():.3f}")

print("\ngated (GLU) forms on two inputs, gate g and value u: output = act(g) * u")
for g, u in [(2.0, 1.0), (2.0, -1.0), (-2.0, 1.0), (0.0, 5.0), (-0.5, 1.0)]:
    print(f"  g {g:>5.1f}, u {u:>5.1f}: SwiGLU {silu(np.array([g]))[0] * u:>7.3f} | GEGLU {gelu_exact(np.array([g]))[0] * u:>7.3f} | ReGLU {relu(np.array([g]))[0] * u:>7.3f}")
