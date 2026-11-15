"""Section 7.4: arithmetic intensity, the roofline, and reading versus writing.  No timing here:
only counting, with published chip specifications (dense 16-bit matrix arithmetic, memory bandwidth).
Run from 04-code/ch07:  python3 chip_arith.py"""
import _common
from tiny.linalg import intensity

CHIPS = {  # name: (trillion operations per second, 16-bit dense matrix units; memory bandwidth, GB/s)
    "L4": (121, 300), "L40S": (362, 864), "A100 80GB": (312, 2039), "H100 SXM": (989, 3350), "B200": (2250, 8000)}
print(f"{'chip':<11}{'trillion ops/s':>15}{'GB/s':>7}{'break-even ops per byte':>25}")
for name, (tops, gbs) in CHIPS.items():
    print(f"{name:<11}{tops:>15,}{gbs:>7,}{tops * 1e12 / (gbs * 1e9):>25.0f}")

h = 4096                                      # width of a mid-sized model's layer
print(f"\nrows @ ({h}, {h}) weights, 16-bit numbers: operations per byte moved")
for rows in [1, 8, 64, 256, 512, 1024, 4096]:
    print(f"  {rows:>5} row(s) at a time: {intensity(rows, h, h):8.1f}")

tops, gbs = CHIPS["H100 SXM"]
for rows in [1, 64, 512]:
    ai = intensity(rows, h, h)
    reach = min(tops, ai * gbs / 1000)
    print(f"H100 SXM, {rows:>3} row(s): at most {reach:7.1f} trillion ops/s ({100 * reach / tops:5.1f}% of the matrix units)")

# A claim that circulates: reading one 32,768-token document into an embedding takes a top chip minutes.
n, d, layers, params = 32768, 768, 12, 110e6   # a 110-million-number embedding model, 12 layers of width 768
weights_ops = 2 * params * n                    # every token meets every weight once
table_ops = 4 * n * n * d * layers              # agreement tables and weighted sums, all layers
total = weights_ops + table_ops
a100 = CHIPS["A100 80GB"][0] * 1e12
print(f"\n32,768-token document: {weights_ops:.2e} (weights) + {table_ops:.2e} (tables) = {total:.2e} operations")
print(f"on an A100 at 40% of its matrix units: {total / (0.4 * a100):.2f} seconds, not minutes")
print(f"at 130 tokens per second (a WRITING speed): {n / 130 / 60:.1f} minutes")
