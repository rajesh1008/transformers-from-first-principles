"""Chapter 20, sections 20.1 to 20.3: a narrow valley in two numbers, f(x, y) = 0.5 (x^2 + 50 y^2), started at (1, 1).
Plain gradient descent, momentum and Adam (with bias correction), each at the best of a small grid of learning rates.
Counts the steps until f < 1e-4, up to 2,000.  No randomness."""
import _common
import json
import numpy as np
from _common import save_json
A = np.array([1.0, 50.0])
f = lambda p: 0.5 * float((A * p * p).sum())
grad = lambda p: A * p
def run(kind, lr, mu=0.9, b1=0.9, b2=0.999, steps=2000, keep=False):
    p = np.array([1.0, 1.0]); m = np.zeros(2); v = np.zeros(2); path = [p.copy()]
    for t in range(1, steps + 1):
        g = grad(p)
        if kind == "sgd":
            p = p - lr * g
        elif kind == "momentum":
            m = mu * m + g; p = p - lr * m
        else:
            m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
            p = p - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8)
        path.append(p.copy())
        if not np.isfinite(f(p)) or f(p) > 1e6:
            return None, path
        if f(p) < 1e-4:
            return t, path
    return steps, path
print("f(x, y) = 0.5 (x^2 + 50 y^2) from (1, 1); steps until f < 1e-4 (2,000 = not reached)")
print("  curvatures 1 and 50; plain gradient descent is stable only for lr < 2/50 = 0.04")
out = {}
for kind, grid in [("sgd", [0.01, 0.02, 0.03, 0.039, 0.04, 0.041]), ("momentum", [0.001, 0.003, 0.01, 0.02, 0.03, 0.04]),
                   ("adam", [0.003, 0.01, 0.03, 0.1, 0.3])]:
    row = []
    for lr in grid:
        n, path = run(kind, lr)
        row.append((lr, n))
        out[f"{kind}@{lr}"] = dict(steps=n, path=[list(map(float, q)) for q in path[:200]])
    print(f"  {kind:9s}" + "  ".join(f"lr {lr:g}: {'diverged' if n is None else n}" for lr, n in row))
# the slow direction: plain GD at its best lr shrinks x by (1 - lr) per step
print(f"  plain GD at lr 0.039: x shrinks by {1 - 0.039:.3f} per step, y by {abs(1 - 0.039 * 50):.3f}; "
      f"steps for x to reach 0.0141 (f = 1e-4 from x alone): {np.log(0.01414) / np.log(0.961):.0f}")
save_json("valley", out)
