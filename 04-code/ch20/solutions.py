"""Chapter 20: numbers for the Check Yourself box (section 20.3) and Exercises 1 to 8 and 10 (Appendix A).  No randomness."""
import _common
import math
import numpy as np
from tiny.optim import warmup_cosine, wsd
print("Check yourself (20.3)")
a = 0.0
for t in range(1, 11):
    a = 0.95 * a + 0.05 * 2
    if t in (1, 10):
        print(f"  (a) after {t} step(s): {a:.4f}; corrected {a / (1 - 0.95 ** t):.4f}")
print(f"  (b) beta 0.98 remembers about 1/(1-0.98) = {1 / (1 - 0.98):.0f} steps")
print(f"  (c) momentum 0.95, steady g: velocity {1 / (1 - 0.95):.0f} g")
a = 0.9 * (0.1 * 4) + 0.1 * (-4)
print(f"  (d) 4 then -4, beta 0.9: uncorrected {a:.4f}; corrected {a / (1 - 0.9 ** 2):.4f}")
print("Exercise 1: valley with curvatures 1 and 200")
print(f"  largest stable lr 2/200 = {2 / 200}; at lr 0.0099 x shrinks by {1 - 0.0099:.4f} per step;"
      f" steps for x from 1 to 0.0141: {math.log(0.01414) / math.log(1 - 0.0099):.0f}")
print(f"Exercise 2: momentum 0.99: steady velocity {1 / (1 - 0.99):.0f} g; window about {1 / (1 - 0.99):.0f} steps")
def adam(G, b1=0.9, b2=0.95):
    m = v = 0.0; out = []
    for t, g in enumerate(G, 1):
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        out.append(m / (1 - b1 ** t) / (math.sqrt(v / (1 - b2 ** t)) + 1e-8))
    return out
print("Exercise 3: Adam step/lr for nudges 1,1,1: " + ", ".join(f"{x:.4f}" for x in adam([1, 1, 1]))
      + "; for 1,-1,1,-1: " + ", ".join(f"{x:.4f}" for x in adam([1, -1, 1, -1])))
print(f"Exercise 4: no bias correction, b2 0.98, first step/lr = 0.1/sqrt(0.02) = {0.1 / math.sqrt(0.02):.4f}")
print(f"Exercise 5: AdamW lr 1e-3, wd 0.1, 10,000 steps: an unpushed number keeps {(1 - 1e-4) ** 10000:.4f} of its size (e^-1 = {math.exp(-1):.4f})")
for vh in (1e-4, 1e-8):
    print(f"Exercise 6: coupled L2 (lambda 0.1, w 1) inside Adam with v_hat {vh:g}: decay part of the step per lr = {0.1 / (math.sqrt(vh) + 1e-8):.4g}"
          f" (decoupled: 0.1)")
n = 70e9
print(f"Exercise 7: 70 billion numbers, 32-bit, weights + nudges + Adam: {n * 16 / 1e12:.2f} TB; Adam's share {n * 8 / 1e12:.2f} TB")
f, w = warmup_cosine(40, 400, 0.1), wsd(40, 400, 0.2)
print(f"Exercise 8: warmup+cosine at step 220: {f(220):.4f}; WSD at step 360: {w(360):.4f}; WSD at step 300: {w(300):.4f}")
A = np.array([1.0, 50.0])
def nesterov(lr, mu=0.9):
    p = np.array([1.0, 1.0]); b = np.zeros(2)
    for t in range(1, 2001):
        g = A * p; b = mu * b + g; p = p - lr * (g + mu * b)
        if 0.5 * (A * p * p).sum() < 1e-4: return t
        if not np.isfinite(p).all() or abs(p).max() > 1e6: return None
    return 2000
print("Exercise 10: Nesterov momentum on the valley, steps to f < 1e-4: " +
      "  ".join(f"lr {lr:g}: {nesterov(lr)}" for lr in (0.003, 0.01, 0.02, 0.03, 0.04)))
