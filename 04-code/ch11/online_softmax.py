"""Chapter 11, sections 2 and 3: softmax one block at a time.  mat's row of scores from Chapter 8's hand
head, computed three ways: the textbook three passes (max, sum, divide), the online two passes (running
max and running sum, rescaled when the max grows), and blocks of two with the output carried along.
Run from 04-code/ch11:  python3 online_softmax.py"""
from _common import *
from tiny.attention import softmax

Q, K, V = X @ W_q, X @ W_k, X @ W_v
s = (K @ Q[5]) / np.sqrt(2)                       # mat's scores against all six seats (causal: all allowed)
print("mat's scores:", r3(s))
ref = softmax(s); out_ref = ref @ V
print("reference shares:", r3(ref), " output:", r3(out_ref))

print("\nthree passes (safe softmax): 1) max  2) sum of exp(s - max)  3) divide")
m = s.max(); l = np.exp(s - m).sum()
print(f"  max {m:.3f} | sum {l:.4f} | shares {r3(np.exp(s - m) / l)}")

print("\nonline: one pass for max and sum together; rescale the sum when the max grows")
m, l = -np.inf, 0.0
for j, x in enumerate(s):
    m_new = max(m, x)
    l = l * np.exp(m - m_new) + np.exp(x - m_new)
    note = "  (max grew: old sum rescaled)" if m_new > m and j > 0 else ""
    m = m_new
    print(f"  seat {j} ({WORDS[j]:>3}): score {x:6.3f} -> running max {m:6.3f}, running sum {l:.4f}{note}")
print(f"  final shares {r3(np.exp(s - m) / l)} equal reference: {bool(np.allclose(np.exp(s - m) / l, ref))}")

print("\nblocks of two seats, output carried along (never holding the whole row of shares):")
m, l, o = -np.inf, 0.0, np.zeros(2)
for b in range(3):
    sb, vb = s[2 * b:2 * b + 2], V[2 * b:2 * b + 2]
    m_new = max(m, sb.max())
    scale = np.exp(m - m_new)
    p = np.exp(sb - m_new)
    l = l * scale + p.sum()
    o = o * scale + p @ vb
    m = m_new
    print(f"  block {b} ({WORDS[2*b]}, {WORDS[2*b+1]}): max {m:6.3f} | rescale {scale:.4f} | sum {l:.4f} | unnormalised output {r3(o)}")
print(f"  output = unnormalised / sum = {r3(o / l)}  equal reference: {bool(np.allclose(o / l, out_ref))}"
      f" | difference {np.abs(o / l - out_ref).max():.1e}")

print("\nwhy subtract the max at all: scores of 100 and 101 in 32-bit and 16-bit")
for dt in (np.float32, np.float16):
    z = np.array([100.0, 101.0], dtype=dt)
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(z) / np.exp(z).sum()
    safe = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    print(f"  {np.dtype(dt).name}: naive {naive} | with the max subtracted {safe.astype(np.float64).round(4)}")
print("  largest x with exp(x) finite: float16", float(np.log(np.finfo(np.float16).max)).__round__(2),
      "| float32", float(np.log(np.finfo(np.float32).max)).__round__(2))
