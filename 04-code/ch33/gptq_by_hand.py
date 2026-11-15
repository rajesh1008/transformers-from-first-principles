"""Chapter 33, 33.6: GPTQ's idea on two weights.  One output = w1 x1 + w2 x2, with x1 and x2 strongly correlated in
the calibration data.  Round both weights to a grid of step 0.2 independently, then round w1 and let w2 absorb the
error before it is rounded.  Mean squared output error over the calibration rows for each.  Seed 1865."""
import numpy as np
rng = np.random.default_rng(1865)
n = 1000
x1 = rng.normal(size=n); x2 = 0.9 * x1 + np.sqrt(1 - 0.81) * rng.normal(size=n)
X = np.stack([x1, x2], 1); w = np.array([0.27, 0.27]); step = 0.2
r = lambda v: np.round(v / step) * step
H = X.T @ X / n
print(f"calibration: {n} rows; correlation of x1 and x2 {np.corrcoef(x1, x2)[0, 1]:.3f}; H = X^T X / n = {np.round(H, 3).tolist()}")
def err(q): return float(np.mean((X @ (w - q)) ** 2))
q_rtn = r(w)
print(f"round each to nearest:   w {w.tolist()} -> {np.round(q_rtn, 3).tolist()}; output mean squared error {err(q_rtn):.5f}")
# GPTQ order: round w1; the error e1 = w1 - q1 moves to w2 by the least-squares update  w2 += e1 * H[0,1] / H[1,1]
# (for two weights this is what GPTQ's inverse-Hessian update reduces to)
q1 = r(w[0]); e1 = w[0] - q1
w2_new = w[1] + e1 * H[0, 1] / H[1, 1]
q_gptq = np.array([q1, r(w2_new)])
print(f"round w1 = {w[0]:.2f} -> {q1:.2f} (error {e1:+.3f}); w2 absorbs it: {w[1]:.3f} + {e1:.3f} x {H[0, 1] / H[1, 1]:.3f} = {w2_new:.3f} -> {r(w2_new):.2f}")
print(f"error-compensated:       w {w.tolist()} -> {np.round(q_gptq, 3).tolist()}; output mean squared error {err(q_gptq):.5f}")
grid = [(a, b) for a in np.arange(-1, 1.01, step) for b in np.arange(-1, 1.01, step)]
best = min(grid, key=lambda q: err(np.array(q)))
print(f"best pair on the grid (searched): {np.round(best, 3).tolist()}; output mean squared error {err(np.array(best)):.5f}")
print(f"weights' own squared error: nearest {float(np.sum((w - q_rtn) ** 2)):.4f}; compensated {float(np.sum((w - q_gptq) ** 2)):.4f}")
