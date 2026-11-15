"""Chapter 20, section 20.4: Adam on one number, by hand.  Five nudges, b1 = 0.9, b2 = 0.95 and 0.999, eps 1e-8.
Prints the running averages, the bias-corrected ones and the step divided by lr.  No randomness."""
import _common
import numpy as np
G = [4.0, 3.0, -1.0, 2.0, 2.5]
for b2 in (0.95, 0.999):
    print(f"b1 0.9, b2 {b2}: nudges {G}")
    print("  t   nudge      m        v       m_hat    v_hat   step/lr (corrected)  step/lr (uncorrected)")
    m = v = 0.0
    for t, g in enumerate(G, 1):
        m = 0.9 * m + 0.1 * g; v = b2 * v + (1 - b2) * g * g
        mh, vh = m / (1 - 0.9 ** t), v / (1 - b2 ** t)
        print(f"  {t}  {g:6.2f}  {m:7.4f}  {v:8.5f}  {mh:7.4f}  {vh:7.4f}     {mh / (np.sqrt(vh) + 1e-8):8.4f}"
              f"            {m / (np.sqrt(v) + 1e-8):8.4f}")
print("same five nudges times 1000 and times 0.001, b2 0.95 (corrected step/lr):")
for k in (1000.0, 0.001):
    m = v = 0.0; s = []
    for t, g in enumerate(G, 1):
        g *= k; m = 0.9 * m + 0.1 * g; v = 0.95 * v + 0.05 * g * g
        s.append(m / (1 - 0.9 ** t) / (np.sqrt(v / (1 - 0.95 ** t)) + 1e-8))
    print(f"  x{k:g}: " + "  ".join(f"{x:.4f}" for x in s))
m = v = 0.0; s = []
for t, g in enumerate(G, 1):
    g *= 1e-9; m = 0.9 * m + 0.1 * g; v = 0.95 * v + 0.05 * g * g
    s.append(m / (1 - 0.9 ** t) / (np.sqrt(v / (1 - 0.95 ** t)) + 1e-8))
print("  x1e-9 (nudges near eps): " + "  ".join(f"{x:.4f}" for x in s))
