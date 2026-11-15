"""Appendix A, Chapter 5: the exercises that need code.  Run: python3 solutions.py"""
import _common
import math
from itertools import permutations
import numpy as np
import _gpt2
from tiny.tokenizer import TOY_VOCAB
from tiny.embed import (TOY_EMBED, TOY_POS, add_rows, add_positions, clock, clock_speeds, dot, cosine,
                        PositionTable, Embedding, embed_with_positions, sinusoidal)

print("Exercise 5.1: 'the dog sat on the hat' + toy positions")
words = "the dog sat on the hat".split()
Y = add_positions(TOY_EMBED[[TOY_VOCAB.index(w) for w in words]], TOY_POS)
print("  ", [f"{w}@{k}={r.tolist()}" for k, (w, r) in enumerate(zip(words, Y))])

print("Exercise 5.2: clock hands for positions 6 and 9:", [round(v, 3) for v in clock(6)], [round(v, 3) for v in clock(9)],
      "; dot(3, 9) =", round(dot(clock(3), clock(9)), 3))

print("Exercise 5.3: orders of 'a cat sat on the hat':", len(set(permutations("a cat sat on the hat".split()))),
      "; of 'the cat sat on the mat':", len(set(permutations("the cat sat on the mat".split()))))

print("Exercise 5.4: position tables of 128 numbers per row:",
      {n: n * 128 for n in (256, 1024, 2048)})

P = _gpt2.tensor("wpe.weight"); E = _gpt2.tensor("wte.weight")
print(f"Exercise 5.5: GPT-2's position table {P.size:,} numbers = {P.size / 124_439_808 * 100:.2f}% of 124,439,808")

def hands(k, speeds):
    out = []
    for s in speeds:
        out += clock(k, s)
    return out
for speeds in ((30.0, 7.5), (30.0, 7.5, 1.875)):
    first = next(k for k in range(1, 5000) if all(abs(u - v) < 1e-9 for u, v in zip(hands(k, speeds), hands(0, speeds))))
    print(f"Exercise 5.6: clocks at {speeds} degrees per token first repeat after {first} tokens")

for scale in (1.0, 0.3, 0.1):
    seen = {}
    for w_id in range(11):
        for k in range(6):
            seen.setdefault(tuple(np.round(add_rows(TOY_EMBED[w_id], TOY_POS[k] * scale), 6)), []).append(1)
    print(f"Exercise 5.9: toy positions x {scale}: arrows shared by two or more word-and-place pairs: "
          f"{sum(1 for v in seen.values() if len(v) > 1)}")

pos = PositionTable(seed=1865)
U = pos.table / np.linalg.norm(pos.table, axis=1, keepdims=True)
C = U @ U.T; np.fill_diagonal(C, -2)
print(f"Exercise 5.10: Tiny's untrained position rows: closest pair has cosine {C.max():.3f}, "
      f"average |cosine| {np.abs(C[C > -2]).mean():.3f}")

for t, name in ((262, "' the'"), (1169, "'the'")):
    print(f"Exercise 5.11: GPT-2 {name}: cosine of input rows at positions 0 and 4 = {cosine(E[t] + P[0], E[t] + P[4]):.3f}; "
          f"at 1 and 5 = {cosine(E[t] + P[1], E[t] + P[5]):.3f}")

for base in (10000.0, 100.0):
    laps = [round(2 * math.pi / s, 1) for s in clock_speeds(12, base)]
    print(f"Exercise 5.14: six clocks with base {base:g}: laps (tokens) {laps}")

def check_positions(ids, context):
    """Exercise 5.12: fail loudly, with a useful message, instead of an IndexError or a silent cut."""
    if len(ids) > context:
        raise ValueError(f"sequence of {len(ids)} tokens is longer than the context length {context}; "
                         f"cut it into windows first")
    return ids
try:
    check_positions(list(range(1025)), 1024)
except ValueError as err:
    print("Exercise 5.12:", err)

for d in (128, 32):
    S = sinusoidal(1100 + 1, d)
    U = S / (d / 2) ** 0.5
    c = U[:1024] @ U[1100]
    print(f"Exercise 5.13: width {d}: position 1100's closest trained row is {int(c.argmax())}, cosine {c.max():.3f}; "
          f"slowest clock laps every {2 * math.pi / clock_speeds(d)[-1]:,.0f} tokens")
