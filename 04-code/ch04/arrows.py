"""Math Interlude: lists of numbers as arrows.  Adding, stretching, length, agreement, angle.
Run from 04-code/ch04:  python3 arrows.py"""
import _common, math
from tiny.embed import dot, length, cosine

cat, mat, sat, on, the = (0, 1), (-1, 1), (1, 1), (1, -1), (1, 0)
add = lambda a, b: tuple(x + y for x, y in zip(a, b))
stretch = lambda k, a: tuple(k * x for x in a)
p = lambda x: f"({x})" if x < 0 else str(x)          # show negative numbers in brackets

print("adding arrows: cat + the =", add(cat, the), "  (same as sat)")
print("stretching: 3 x sat =", stretch(3, sat), "  -1 x on =", stretch(-1, on))
print(f"length of sat = sqrt(1*1 + 1*1) = {length(sat):.4f}")
print(f"length of 3 x sat = {length(stretch(3, sat)):.4f}  (three times as long)")
print()
for name, a, b in [("sat . cat", sat, cat), ("sat . the", sat, the), ("sat . on", sat, on),
                   ("sat . mat", sat, mat), ("on . mat", on, mat), ("sat . sat", sat, sat)]:
    print(f"  {name:<10} = {p(a[0])}*{p(b[0])} + {p(a[1])}*{p(b[1])} = {dot(a, b):>2}   "
          f"cosine {cosine(a, b):>6.3f}   angle {math.degrees(math.acos(round(cosine(a, b), 12))):>5.1f} degrees")
print()
print(f"three numbers each: (1, 2, 3) . (4, -5, 6) = {dot((1, 2, 3), (4, -5, 6))}")
