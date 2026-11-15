"""How far does each of Tiny's 64 clocks turn inside its 1,024-token window, and what is new past it?
Run from 04-code/ch06:  python3 turns.py"""
import _common, math
from tiny.embed import clock_speeds, TINY_CONTEXT

speeds = clock_speeds(128)
turns = [s * (TINY_CONTEXT - 1) / (2 * math.pi) for s in speeds]
full = sum(t >= 1 for t in turns)
print(f"clock 0 turns {turns[0]:.1f} times in {TINY_CONTEXT:,} tokens; clock 63 turns {turns[-1]:.4f} times "
      f"({math.degrees(speeds[-1] * (TINY_CONTEXT - 1)):.1f} degrees)")
print(f"clocks that go all the way round inside the window: {full} of 64; clocks that never do: {64 - full}")
print(f"first clock that never completes a turn: clock {full} (one turn every {2 * math.pi / speeds[full]:,.0f} tokens)")
for pos in [1023, 2047, 4095]:
    a = math.degrees(speeds[-1] * pos)
    print(f"slowest clock at a gap of {pos:>5,}: {a:5.1f} degrees  (largest seen in training, at a gap of 1,023: "
          f"{math.degrees(speeds[-1] * 1023):.1f})")
print(f"slowest clock comes back to 12 o'clock after {2 * math.pi / speeds[-1]:,.0f} tokens")
