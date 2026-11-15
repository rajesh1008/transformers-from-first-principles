"""Chapter 13, Interlude: average and spread, and normalising a list.  A made-up list of eight numbers
chosen so the arithmetic comes out whole, then the same steps on a list ten times larger and shifted.
Run from 04-code/ch13:  python3 interlude.py"""
from _common import *
from tiny.norm import average, spread, layer_norm

L = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
m = L.mean()
dev = L - m
print("the list:", r3(L))
print(f"average: sum {L.sum():.0f} / {len(L)} = {m:.3f}")
print("distances from the average:", r3(dev), "| they always sum to", round(float(dev.sum()), 12))
print("squared distances:", r3(dev ** 2), f"| their average {np.mean(dev ** 2):.3f}")
s = np.sqrt(np.mean(dev ** 2))
print(f"spread = square root of that average = {s:.3f}")
z = dev / s
print("normalised list (distance / spread):", r3(z))
print(f"  its average {z.mean():.3f}, its spread {np.sqrt(np.mean((z - z.mean()) ** 2)):.3f}")
print(f"  with eps = 1e-5 inside the root (what the code does): {r3(layer_norm(L), 4)}")
L2 = 10 * L + 3
print("\nthe list times 10, plus 3:", r3(L2))
print(f"  average {L2.mean():.3f}, spread {float(spread(L2)[0]):.3f}")
print("  normalised:", r3(layer_norm(L2), 4), "| same as before:", bool(np.allclose(layer_norm(L2), layer_norm(L), atol=1e-4)))
print("\nwhat normalising keeps and what it throws away:")
print("  keeps: the order of the numbers and how far apart they are compared with the spread")
print("  throws away: the average (3 or 53) and the overall size (spread 2 or 20)")
print("\nthe spread with d - 1 instead of d (the 'sample' spread some libraries default to):",
      f"{np.std(L, ddof=1):.4f} against {np.std(L):.4f}; ratio {np.std(L, ddof=1) / np.std(L):.4f} = sqrt(8/7)")
