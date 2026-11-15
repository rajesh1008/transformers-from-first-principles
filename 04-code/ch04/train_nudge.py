"""Where the numbers come from: one nudge by hand, then 16 numbers per word learned from the novel.
Run from 04-code/ch04:  python3 train_nudge.py"""
import _common, time
import numpy as np
from nudge import nudge, squash, corpus, neighbour_pairs, train, nearest

cat, mat = np.array([0.0, 1.0]), np.array([-1.0, 1.0])
new, dot, sure = nudge(cat, mat, together=True, step=0.1)
print(f"one nudge by hand: cat {cat.tolist()} . mat {mat.tolist()} = {dot:.0f}; squash({dot:.0f}) = {sure:.3f}")
print(f"  pull cat toward mat by 0.1 x (1 - {sure:.3f}) x mat -> cat = {np.round(new, 3).tolist()}")
print(f"  new agreement {np.dot(new, mat):.3f} (was {dot:.0f})")
print(f"  squash at -4, 0, 4: {squash(-4):.3f}, {squash(0):.3f}, {squash(4):.3f}")

w, counts, vocab, index, ids = corpus()
P = neighbour_pairs(ids)
print(f"\nthe novel: {len(w):,} words; {len(vocab)} different words seen 5 times or more; "
      f"{len(P):,} (word, neighbour) pairs within 2 places")
print(f"  those 699 words make up {sum(counts[x] for x in vocab) / len(w) * 100:.0f}% of the novel; times seen: "
      + ", ".join(f"{x} {counts[x]}" for x in ["the", "said", "alice", "queen", "king", "cat", "tea", "dog"]))
track = [("said", "replied"), ("said", "tea"), ("queen", "king"), ("two", "three")]
t0 = time.perf_counter()
A, vocab, index, hist = train(d=16, epochs=10, seed=1865, track=track)
print(f"trained 16 numbers per word, 10 passes, seed 1865, in {time.perf_counter() - t0:.1f} s")
print("cosine before training -> after:")
for (x, y), a, b in zip(track, hist[0], hist[-1]):
    print(f"  {x:<6} {y:<8} {a:>6.2f} -> {b:>5.2f}")
print("nearest neighbours after training (cosine):")
for q in ["said", "queen", "two", "head", "looked", "alice", "cat"]:
    print(f"  {q:<7}", ", ".join(f"{x} {c:.2f}" for x, c in nearest(A, vocab, index, q, 5)))
print(f"row for 'cat' (first 4 of 16): {np.round(A[index['cat']][:4], 2).tolist()}")
