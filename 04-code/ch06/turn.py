"""Turning one arrow by its position: cat's row (0, 1) at positions 0 to 5, 30 degrees per token.
Run from 04-code/ch06:  python3 turn.py"""
import _common, math
from tiny.embed import rotate_pair, length, clock

cat = (0.0, 1.0)
for pos in range(6):
    a, u = rotate_pair(cat[0], cat[1], math.radians(30 * pos))
    print(f"cat at position {pos}: turned {30 * pos:3d} degrees -> ({a:6.3f}, {u:6.3f})  length {length((a, u)):.3f}")
print("same as Chapter 5's clock hand at position 5:", [round(x, 3) for x in clock(5)])
sat = (1.0, 1.0)
a, u = rotate_pair(sat[0], sat[1], math.radians(60))
print(f"sat (1, 1) turned 60 degrees: ({a:.3f}, {u:.3f}); length before {length(sat):.3f}, after {length((a, u)):.3f}")
