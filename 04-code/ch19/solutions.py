"""Numbers for Appendix A, Chapter 19: the Check Yourself box and the exercises."""
import _common
import numpy as np
from tiny.model import Tiny
from tiny.init import initialise
from _common import load_json, novel_ids, tok, SEED, probe, rms
rng = np.random.default_rng(SEED)
print("Check Yourself")
x = rng.normal(0, 1, size=(100_000, 100))
print(f"  (a) sum of 100 numbers of spread 1: rule 10, measured {x.sum(1).std():.3f}")
print(f"  (b) their average: rule 0.1, measured {x.mean(1).std():.4f}")
print(f"  (c) fan-in 512 at 0.02: {0.02 * np.sqrt(512):.4f}; spread for 1: 1/sqrt(512) = {1 / np.sqrt(512):.4f}")
a = rng.normal(0, 3, 100_000); b = rng.normal(0, 4, 100_000)
print(f"  (d) independent spreads 3 and 4 added: rule 5, measured {(a + b).std():.3f}")
print("\nExercises")
print(f"  1. 64 numbers of spread 0.5: {0.5 * 8:.1f}")
print(f"  3. ReLU keeps half the mean square: spread 1/sqrt(2) = {1/np.sqrt(2):.4f}; He's spread for fan-in 128: sqrt(2/128) = {np.sqrt(2/128):.4f}")
print(f"  4. Tiny's residual writers under the scaled start: 0.02/sqrt(24) = {0.02/np.sqrt(24):.5f}")
TL = load_json("tiny_layers")["gpt2x1"]
pred = np.sqrt(TL["stream"][0] ** 2 + sum(a ** 2 + f ** 2 for a, f in zip(TL["att"], TL["ff"])))
print(f"  5. stream leaving layer 11 predicted from the 24 branch spreads as independent: {pred:.4f}; measured {TL['stream'][-1]:.4f}")
g = np.array([3.0, 4.0, 12.0])
print(f"  6. nudges 3, 4, 12: global norm {np.sqrt((g ** 2).sum()):.0f}; clip at 1 multiplies by {1/np.sqrt((g ** 2).sum()):.4f}: {np.round(g / 13, 4).tolist()}")
print(f"  7. row spread 0.001, eps 1e-5: 1/sqrt(1e-6 + 1e-5) = {1/np.sqrt(1e-6 + 1e-5):.1f}")
LG = load_json("logits")
for s, k in ((0.02, "1"), (0.2, "10")):
    print(f"  8. score spread for tables of spread {s}: rule 128 s^2 = {128 * s * s:.4f}; measured {LG[k]['plain'][0]:.4f}")
T = tok(); nov = np.array(novel_ids(T))
starts = np.random.default_rng(SEED).integers(0, len(nov) - 129, size=8)
B = np.stack([nov[a:a + 129] for a in starts])
P = probe(initialise(Tiny(), "fan_in", np.sqrt(2)), B)
print(f"  9. He's start (fan-in times sqrt(2)): stream leaving layer 11 {P['stream'][-1]:.4f}; layer-0 entropy {P['entropy'][0]:.3f}")
print("  10. layer-0 attention entropy against the scale multiplier:")
for k in (1, 2, 3, 4, 5, 6, 8, 10):
    P = probe(initialise(Tiny(n_layers=1), "gpt2", k), B[:2])
    print(f"      x{k:<3d} entropy {P['entropy'][0]:.3f}")
