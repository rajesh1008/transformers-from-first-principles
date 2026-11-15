"""Chapter 20, section 20.9: what each optimiser stores, per learned number and in total, in 32-bit.
Counts only; no timing."""
import _common
MODELS = [("Tiny", 2_766_592), ("GPT-2 small", 124_439_808), ("SmolLM2-135M", 134_515_008), ("Qwen3-0.6B", 596_049_920),
          ("a 7-billion-number model", 7_000_000_000)]
OPT = [("SGD", 0), ("SGD + momentum", 1), ("Adam / AdamW", 2), ("Muon (one buffer on tables, Adam's two elsewhere)", 1),
       ("Adafactor-style (row and column averages)", 0)]
print("extra numbers stored per learned number: SGD 0, momentum 1, Adam 2, Muon about 1")
print(f"{'model':28s} {'learned':>15s} {'weights MB':>12s} {'+ nudges MB':>12s} {'+ Adam MB':>12s} {'total MB':>12s}")
for name, n in MODELS:
    w = n * 4 / 1e6
    print(f"{name:28s} {n:15,d} {w:12,.1f} {w:12,.1f} {2 * w:12,.1f} {4 * w:12,.1f}")
# Tiny exact: Muon share
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tiny.model import Tiny
m = Tiny(); t = m.tables()
inside = sum(v.size for k, v in t.items() if v.ndim == 2 and k not in ("E", "P"))
other = sum(v.size for k, v in t.items()) - inside
print(f"Tiny: tables inside the layers with two axes {inside:,}; E, P, gains and biases {other:,}")
print(f"  Muon state: {inside:,} + 2 x {other:,} = {inside + 2 * other:,} numbers ({(inside + 2 * other) * 4 / 1e6:.1f} MB);"
      f" Adam state {2 * (inside + other):,} ({2 * (inside + other) * 4 / 1e6:.1f} MB)")
