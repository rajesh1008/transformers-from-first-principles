"""Stretching Tiny's 1,024-token window to 4,096: do nothing, squeeze the positions, or slow the clocks.
Run from 04-code/ch06:  python3 scaling.py"""
import _common, math
from tiny.embed import clock_speeds

d, train, target = 128, 1024, 4096
s = target / train
plain = clock_speeds(d)
squeezed = [v / s for v in plain]                              # position interpolation: position p -> p / 4
new_base = 10000 * s ** (d / (d - 2))                          # the "base change" rule
slowed = clock_speeds(d, new_base)
def full_turn(i):    # did clock i go all the way round inside the training window? then every angle was seen
    return plain[i] * (train - 1) >= 2 * math.pi
print(f"stretch factor {s:.0f}; new base for the base change: {new_base:,.0f}")
for name, sp in [("do nothing", plain), ("squeeze positions", squeezed), ("slow the clocks", slowed)]:
    new = sum(1 for i, v in enumerate(sp) if not full_turn(i) and v * (target - 1) > 1.01 * plain[i] * (train - 1))
    print(f"{name:>17}: fastest clock {math.degrees(sp[0]):5.1f} deg/token, slowest {math.degrees(sp[-1]) * 1000:6.3f} "
          f"millidegrees/token; neighbours on the fastest clock agree {math.cos(sp[0]):.3f}; "
          f"clocks going over 1% past their trained angle by gap {target - 1:,}: {new}")
print("speed after / speed before, clock by clock (0, 16, 32, 48, 63):")
for name, sp in [("squeeze positions", squeezed), ("slow the clocks", slowed)]:
    print(f"  {name:>17}:", [round(sp[i] / plain[i], 3) for i in [0, 16, 32, 48, 63]])
