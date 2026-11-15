"""Section 5.4: positions as clock hands.  Run: python3 clocks.py"""
import _common
import math
from tiny.embed import clock, clock_speeds, dot, length

print("one clock, 30 degrees per token (one 'hour' per token), hand starts at 12 o'clock:")
for k in range(13):
    x, y = clock(k)
    print(f"  position {k:>2}: hand at {k % 12 or 12:>2} o'clock -> (across, up) = ({x:6.3f}, {y:6.3f}), length {length((x, y)):.3f}")
print("position 12 equals position 0:", [round(v, 9) for v in clock(12)] == [round(v, 9) for v in clock(0)])

print("agreement (dot product) between hands, by gap:")
for a, b in ((0, 4), (1, 5), (2, 6), (7, 11), (0, 1), (5, 6), (0, 3), (0, 6)):
    print(f"  positions {a:>2} and {b:>2} (gap {b - a}): {dot(clock(a), clock(b)):6.3f}")

def two_clocks(k):
    return clock(k, 30.0) + clock(k, 7.5)       # a fast hand and a slow hand: four numbers
print("two clocks (30 and 7.5 degrees per token), position 0:", [round(v, 3) for v in two_clocks(0)])
print("two clocks, position 12:", [round(v, 3) for v in two_clocks(12)], "-> same as 0?",
      [round(v, 6) for v in two_clocks(12)] == [round(v, 6) for v in two_clocks(0)])
print("two clocks repeat only after", next(k for k in range(1, 1000)
      if all(abs(u - v) < 1e-9 for u, v in zip(two_clocks(k), two_clocks(0)))), "tokens")

for d in (12, 128):
    speeds = clock_speeds(d)
    laps = [2 * math.pi / s for s in speeds]
    print(f"d = {d}: {len(speeds)} clocks; turn per token, fastest {math.degrees(speeds[0]):.1f} deg, "
          f"slowest {math.degrees(speeds[-1]):.4f} deg; one lap takes {laps[0]:.2f} tokens (fastest) "
          f"to {laps[-1]:,.0f} tokens (slowest)")
    if d == 12:
        print("   all six laps (tokens):", [round(x, 1) for x in laps])
        print("   each clock is slower than the one before by a factor of", round(speeds[0] / speeds[1], 3))
