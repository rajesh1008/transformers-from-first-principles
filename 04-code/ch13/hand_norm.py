"""Chapter 13, sections 2 and 3: layer normalisation and RMSNorm by hand on sat's row.
Toy rows: the running sentence after Chapter 12's hand layer (Chapter 8's head, then the 2 -> 3 -> 2
block, each added to the row).  Then Tiny's 128-number row for ' sat' after Chapter 12's untrained layer.
Run from 04-code/ch13:  python3 hand_norm.py"""
from _common import *
from tiny.norm import layer_norm, rms_norm, average, spread, rms

A, Y = hand_layer(X)
print("the running sentence after Chapter 12's hand layer (no normalisation yet):")
for w, y in zip(WORDS, Y):
    print(f"  {w:>3}: {r3(y)}  length {np.linalg.norm(y):.3f}")
sat = Y[2]
m = sat.mean(); d = sat - m; s = np.sqrt(np.mean(d ** 2))
print(f"\nsat's row {r3(sat)}: average {m:.3f}, distances {r3(d)}, spread {s:.3f}")
print(f"  layer norm (no gain, no bias): {r3(d / s)}; with eps 1e-5: {r3(layer_norm(sat), 6)}")
q = np.sqrt(np.mean(sat ** 2))
print(f"  RMSNorm: root mean square {q:.3f}; {r3(sat)} / {q:.3f} = {r3(sat / q)}")
print(f"  lengths: before {np.linalg.norm(sat):.3f}, after layer norm {np.linalg.norm(layer_norm(sat)):.3f}, "
      f"after RMSNorm {np.linalg.norm(rms_norm(sat)):.3f} (sqrt(2) = {np.sqrt(2):.3f})")
print("\nevery toy row through layer norm (two numbers: only which one is bigger survives):")
for w, y in zip(WORDS, Y):
    print(f"  {w:>3}: {r3(y)} -> {r3(layer_norm(y), 3)} | RMSNorm -> {r3(rms_norm(y))}")
print("\na flat row (3, 3): spread 0")
with np.errstate(invalid="ignore", divide="ignore"):
    flat = np.array([3.0, 3.0])
    print("  without eps:", r3((flat - flat.mean()) / np.sqrt(np.mean((flat - flat.mean()) ** 2))), "| with eps:", r3(layer_norm(flat)))
g = np.array([2.0, 0.5]); b = np.array([0.0, 1.0])
print(f"\ngain {r3(g)} and bias {r3(b)} after layer norm: sat -> {r3(layer_norm(sat, g, b), 3)}")
print(f"  gain on RMSNorm: sat -> {r3(rms_norm(sat, g))}")

# Tiny: sat's 128-number row after Chapter 12's untrained layer
from tiny.attention import MultiHead
from tiny.ffn import FeedForward
tok, ids, Xs = tiny_rows()
att, ff = MultiHead(n_heads=4), FeedForward()
A1 = Xs + att(Xs); Y1 = A1 + ff(A1)
pieces = [tok.decode([i]) for i in ids]
k = pieces.index(" sat")
row = Y1[k]
print(f"\nTiny, {len(ids)} tokens {pieces}; ' sat' is token {k}")
print(f"  ' sat' after Chapter 12's layer: average {row.mean():.5f}, spread {row.std():.5f}, root mean square {np.sqrt(np.mean(row ** 2)):.5f}, length {np.linalg.norm(row):.4f}")
print(f"  first four numbers: {r3(row[:4], 4)}")
z = layer_norm(row)
print(f"  after layer norm: first four {r3(z[:4], 3)}; average {z.mean():.6f}, spread {z.std():.6f}, length {np.linalg.norm(z):.3f} (sqrt(128) = {np.sqrt(128):.3f})")
zr = rms_norm(row)
print(f"  after RMSNorm: first four {r3(zr[:4], 3)}; length {np.linalg.norm(zr):.3f}")
print(f"  layer norm against RMSNorm, largest difference: {np.abs(z - zr).max():.4f} (the row's average is {row.mean() / row.std():.3f} spreads from zero)")
print(f"  every row's length after layer norm: {r3(np.linalg.norm(layer_norm(Y1), axis=1), 3)}")
print(f"  every row's length before: {r3(np.linalg.norm(Y1, axis=1), 4)}")
