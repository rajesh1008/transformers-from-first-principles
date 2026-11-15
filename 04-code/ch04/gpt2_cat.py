"""Embeddings in GPT-2: 768 numbers for ' cat', its neighbours, and the shape of the whole table.
Run from 04-code/ch04:  python3 gpt2_cat.py   (downloads GPT-2 small's weights once, 548 MB)"""
import _common, _gpt2
from _common import SENTENCE
import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")
W = _gpt2.tensor("wte.weight").astype(float)       # the token embedding table
print(f"GPT-2's table: {W.shape[0]:,} rows x {W.shape[1]} numbers = {W.size:,} numbers, "
      f"{W.size * 4 / 1e6:.0f} MB at 4 bytes each")
ids = enc.encode(SENTENCE)
X = W[ids]
print(f"{SENTENCE!r} -> ids {ids} -> {X.shape[0]} rows of {X.shape[1]} numbers")
cat = W[enc.encode(" cat")[0]]
print(f"\nrow 3797 (' cat'), first 8 of 768: {np.round(cat[:8], 3).tolist()}")
print(f"  smallest {cat.min():.3f}, largest {cat.max():.3f}, average {cat.mean():.4f}")
print(f"  {np.mean(np.abs(cat) < 0.2) * 100:.0f}% of its numbers lie between -0.2 and 0.2")
print(f"  length sqrt(sum of squares) = {np.linalg.norm(cat):.3f}")
big = np.argsort(-np.abs(cat))[:3]
print(f"  its three biggest numbers sit at places {big.tolist()}: {np.round(cat[big], 3).tolist()}")

N = W / np.linalg.norm(W, axis=1, keepdims=True)
def nearest(tok, k=6):
    i = enc.encode(tok)[0]
    s = N @ N[i]
    return [(enc.decode([j]), round(float(s[j]), 2)) for j in np.argsort(-s)[1:k + 1]]
print()
for t in [" cat", " sat", " mat", " the"]:
    print(f"nearest to {t!r}: {nearest(t)}")
print()
for t in [" cat", "cat", " Cat", " cats", " dog", " mat", " the"]:
    print(f"  cosine(' cat', {t!r:>7}) = {float(N[3797] @ N[enc.encode(t)[0]]):.2f}")

L = np.linalg.norm(W, axis=1)
print(f"\nrow lengths: shortest {L.min():.2f}, average {L.mean():.2f}, longest {L.max():.2f}")
o = np.argsort(L)
print("ten shortest rows:", [enc.decode([i]) for i in o[:10]])
print("ten longest rows: ", [enc.decode([i]) for i in o[-10:]])
mean_dir = W.mean(0) / np.linalg.norm(W.mean(0))
rng = np.random.default_rng(1865)
a, b = rng.integers(0, len(W), 20000), rng.integers(0, len(W), 20000)
print(f"cosine of each row with the table's average direction: {float((N @ mean_dir).mean()):.2f} on average")
print(f"cosine between 20,000 random pairs of rows: {float((N[a] * N[b]).sum(1).mean()):.2f} on average "
      f"(0 would mean no shared lean)")

print()
for t in [" said", " queen"]:
    print(f"GPT-2's nearest to {t!r}: {nearest(t, 5)}")
print(f"nearest to ' bank': {nearest(' bank', 6)}")
for t in [" river", " shore", " money", " loan", " cat"]:
    print(f"  cosine(' bank', {t!r:>8}) = {float(N[enc.encode(' bank')[0]] @ N[enc.encode(t)[0]]):.2f}")
