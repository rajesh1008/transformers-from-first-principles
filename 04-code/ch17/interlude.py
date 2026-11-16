"""Chapter 17 Maths Interlude: slope as 'which way to nudge', measured by wiggling and compared with
the rule for each function; and the chain rule, 'nudges multiply', checked on a chain of two.
Plain Python and the math module; nothing random."""
import math

def wiggle(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

print("slope by wiggling (h = 0.000001) against the rule")
cases = [
    ("3x + 1", lambda x: 3 * x + 1, lambda x: 3.0, [-2.0, 0.0, 5.0]),
    ("x squared", lambda x: x * x, lambda x: 2 * x, [-1.5, 0.0, 3.0]),
    ("e to the x", math.exp, math.exp, [-1.0, 0.0, 2.0]),
    ("ln x", math.log, lambda x: 1 / x, [0.5, 1.0, 4.0]),
    ("max(0, x)", lambda x: max(0.0, x), lambda x: 1.0 if x > 0 else 0.0, [-1.0, 0.5, 2.0]),
]
for name, f, rule, xs in cases:
    for x in xs:
        print(f"  {name:<11} at x = {x:+.1f}: value {f(x):+.4f}  wiggled slope {wiggle(f, x):+.4f}  rule {rule(x):+.4f}")

print("\nthe chain rule: nudges multiply")
g = lambda x: 3 * x + 1          # inner
f = lambda u: u * u              # outer
x = 2.0; u = g(x)
print(f"  x = {x}: u = 3x + 1 = {u}; y = u squared = {f(u)}")
print(f"  slope of u in x: {wiggle(g, x):.4f}; slope of y in u: {wiggle(f, u):.4f}; product {wiggle(g, x) * wiggle(f, u):.4f}")
print(f"  slope of y in x, wiggled straight through: {wiggle(lambda t: f(g(t)), x):.4f}")
x = 1.0; inner = lambda t: t * t; outer = math.exp
print(f"  e to the (x squared) at x = 1: rule e^1 x 2 = {math.e * 2:.4f}; wiggled {wiggle(lambda t: outer(inner(t)), x):.4f}")
print("\nthe slope of a surprise: -ln(share), share = e^s / (e^s + rest)")
s, rest = 0.5, 3.0
share = math.exp(s) / (math.exp(s) + rest)
sur = lambda t: -(t - math.log(math.exp(t) + rest))
print(f"  s = {s}, rest = {rest}: share {share:.4f}; slope of surprise in s, wiggled {wiggle(sur, s):+.4f}; rule share - 1 = {share - 1:+.4f}")
other = lambda t: -(s - math.log(math.exp(s) + math.exp(t) + rest - 1))   # surprise of the true word as a rival's score t moves
t0 = 0.0; share_rival = math.exp(t0) / (math.exp(s) + math.exp(t0) + rest - 1)
print(f"  a rival's score t = {t0}: its share {share_rival:.4f}; slope of the true word's surprise in t, wiggled {wiggle(other, t0):+.4f}; rule = the rival's share")
print("\nCheck Yourself (answers in Appendix A)")
print(f"  1. slope of 5 - 2x at x = 7: {wiggle(lambda x: 5 - 2 * x, 7.0):+.4f}")
print(f"  2. slope of (2x - 1) squared at x = 3: {wiggle(lambda x: (2 * x - 1) ** 2, 3.0):+.4f} (rule 2 x 5 x 2 = 20)")
print(f"  3. slope of ln(1 + e^x) at x = 0: {wiggle(lambda x: math.log(1 + math.exp(x)), 0.0):+.4f} (rule 1/(1+1) x 1 = 0.5)")
