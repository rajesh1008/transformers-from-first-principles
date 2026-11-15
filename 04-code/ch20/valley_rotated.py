"""Chapter 20, section 20.5: the same valley turned by 45 degrees, f = 0.5 p^T R^T diag(1, 50) R p, from the point
that is (1, 1) in the valley's own axes.  Adam's per-number step sizes only help when the valley's axes are the
numbers' axes.  Same grids as valley.py; steps until f < 1e-4, up to 2,000.  No randomness."""
import _common
import numpy as np
from _common import save_json
th = np.pi / 4
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
H = R.T @ np.diag([1.0, 50.0]) @ R
p0 = R.T @ np.array([1.0, 1.0])
f = lambda p: 0.5 * float(p @ H @ p)
def run(kind, lr, steps=2000):
    p = p0.copy(); m = np.zeros(2); v = np.zeros(2); path = [p.copy()]
    for t in range(1, steps + 1):
        g = H @ p
        if kind == "sgd":
            p = p - lr * g
        elif kind == "momentum":
            m = 0.9 * m + g; p = p - lr * m
        else:
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            p = p - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        path.append(p.copy())
        if not np.isfinite(f(p)) or f(p) > 1e6:
            return None, path
        if f(p) < 1e-4:
            return t, path
    return steps, path
print("the valley of valley.py turned by 45 degrees; same start in the valley's own axes; steps until f < 1e-4")
out = {}
for kind, grid in [("sgd", [0.01, 0.02, 0.03, 0.039, 0.04, 0.041]), ("momentum", [0.001, 0.003, 0.01, 0.02, 0.03, 0.04]),
                   ("adam", [0.003, 0.01, 0.03, 0.1, 0.3])]:
    row = []
    for lr in grid:
        n, path = run(kind, lr); row.append((lr, n))
        out[f"{kind}@{lr}"] = dict(steps=n, path=[list(map(float, q)) for q in path[:200]])
    print(f"  {kind:9s}" + "  ".join(f"lr {lr:g}: {'diverged' if n is None else n}" for lr, n in row))
save_json("valley_rotated", out)
