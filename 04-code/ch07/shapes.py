"""Section 7.2: shapes as bookkeeping.  The shape rule, and what each product costs.
Run from 04-code/ch07:  python3 shapes.py"""
import _common
from tiny.linalg import check_shapes, work

cases = [
    ("toy table: X @ X.T",                   (6, 2),       (2, 6)),
    ("columns with columns: X.T @ X",        (2, 6),       (6, 2)),
    ("one dot product: row @ column",        (1, 2),       (2, 1)),
    ("Tiny's sentence: block @ block.T",     (7, 128),     (128, 7)),
    ("Tiny's full window: W @ W.T",          (1024, 128),  (128, 1024)),
    ("one token scored against 2,048 rows",  (1, 128),     (128, 2048)),
    ("a window scored against 2,048 rows",   (1024, 128),  (128, 2048)),
]
print(f"{'product':<38}{'shapes':>26}{'answer':>14}{'multiplications':>17}{'additions':>14}")
for name, a, b in cases:
    out = check_shapes(a, b)
    mul, add = work(a[0], a[1], b[1])
    print(f"{name:<38}{str(a) + ' @ ' + str(b):>26}{str(out):>14}{mul:>17,}{add:>14,}")

n, d = 1024, 128
mul, add = work(n, d, n)
print(f"\nfull window: {n * n:,} dot products of {d} numbers = {mul + add:,} operations "
      f"(about 2 x {n} x {n} x {d} = {2 * n * n * d:,})")
for bad in [((6, 2), (6, 2)), ((7, 128), (7, 128))]:
    try:
        check_shapes(*bad)
    except ValueError as e:
        print("refused:", e)
