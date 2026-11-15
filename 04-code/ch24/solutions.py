"""Appendix A, Chapter 24: Check Yourself and Exercises 1 to 7 (by hand), computed.  python3 solutions.py"""
import _common
import math, json, os
from _common import HERE
print("Check yourself: probabilities 0.5, 0.25, 0.125, 0.5")
ps = [0.5, 0.25, 0.125, 0.5]
L = -sum(math.log(p) for p in ps) / 4
print(f"  loss {L:.4f} nats, perplexity {math.exp(L):.4f}")
ps2 = [0.5, 0.25, 0.0001, 0.5]
L2 = -sum(math.log(p) for p in ps2) / 4
print(f"  with 0.0001 as the third: loss {L2:.4f} nats, perplexity {math.exp(L2):.2f}")
print("1. p = 0.2 every step: loss %.4f nats, %.4f bits, perplexity %.1f" % (-math.log(.2), -math.log2(.2), 1 / .2))
print("2. uniform: ln 2048 = %.4f, ln 50257 = %.4f" % (math.log(2048), math.log(50257)))
p3 = 3001 / (393597 + 2048)
print("3. unigram with add-one: p = 3001 / 395645 = %.6f; surprise %.4f" % (p3, -math.log(p3)))
print("4. branch at b decays over b/4 (a fifth of the finished length b/0.8): lengths 1,000, 2,000, 5,000 need b = 800, 1,600, 4,000;"
      " job = 4,000 + 200 + 400 + 1,000 = %d steps" % (4000 + 200 + 400 + 1000))
lab = open(os.path.join(HERE, "outputs", "lab24.txt")).read() if os.path.exists(os.path.join(HERE, "outputs", "lab24.txt")) else ""
print("5. see Lab 24 Breakage 3 for the spread of 20 draws; a difference below about twice that spread is not reported")
n = math.ceil(2 * (0.25 / 0.025) ** 2)
print(f"6. spread of a difference of two means of n seeds = 0.25 x sqrt(2/n); for it to be at most half of 0.05: n >= {n}")
print("7. clip at 1.0: norms of 0.8 are untouched.  Clip at 0.5: every step's nudges scaled by 0.5/0.8 = 0.625; Adam divides by the"
      " running size of the nudges, so a constant scale on every step changes its steps hardly at all (only through eps)")
