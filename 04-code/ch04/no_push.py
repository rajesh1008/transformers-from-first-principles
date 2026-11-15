"""Why the nudge rule needs its push: the same training with no random partners.
Run from 04-code/ch04:  python3 no_push.py"""
import _common
import numpy as np
from nudge import train, nearest

rng = np.random.default_rng(1865)
for negatives in (5, 0):
    A, vocab, index = train(d=16, epochs=10, seed=1865, negatives=negatives)
    N = A / np.linalg.norm(A, axis=1, keepdims=True)
    a, b = rng.integers(0, len(A), 5000), rng.integers(0, len(A), 5000)
    print(f"{negatives} random partners per real pair: average cosine of 5,000 random pairs of words "
          f"{float((N[a] * N[b]).sum(1).mean()):.2f}; cos(said, tea) {float(N[index['said']] @ N[index['tea']]):.2f}; "
          f"nearest to said {nearest(A, vocab, index, 'said', 3)}")
