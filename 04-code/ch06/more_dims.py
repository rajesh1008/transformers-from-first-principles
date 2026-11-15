"""More dimensions: every pair of numbers gets its own clock.  First a row of 8 (4 clocks), then Tiny's 128.
Run from 04-code/ch06:  python3 more_dims.py"""
import _common, math
import numpy as np
from _common import SENTENCE, TINY_BPE
from tiny.tokenizer import load_bpe, TINY_VOCAB_SIZE
from tiny.embed import Embedding, Rotary, clock_speeds, apply_rotary, dot, length

speeds = clock_speeds(8)                           # four clocks for a row of 8 numbers
print("four clocks, radians per token:", [round(s, 4) for s in speeds])
print("  degrees per token:            ", [round(math.degrees(s), 2) for s in speeds])
q = np.array([[1.0, 0.0, 0.5, 0.5, -1.0, 1.0, 0.0, 2.0]])
for pos in [0, 1, 2]:
    print(f"  row turned for position {pos}:", apply_rotary(q, [pos], speeds=speeds)[0].round(3).tolist())

tok = load_bpe(TINY_BPE)
ids = tok.encode(SENTENCE)                         # [363, 725, 973, 327, 262, 283, 303]
emb = Embedding(TINY_VOCAB_SIZE, seed=1865)
the, cat = emb([ids[0]])[0], emb([ids[1]])[0]
rot = Rotary()                                     # 64 clocks, base 10,000
print(f"\nTiny: {rot.d} numbers, {len(rot.speeds)} clocks; plain agreement of 'the' and ' cat' rows: {dot(the, cat):.6f}")
for m in [0, 1, 100, 1000, 5000]:
    a = rot([the], start=m)[0]; b = rot([cat], start=m + 1)[0]
    print(f"  the@{m:<4} . cat@{m + 1:<4} = {dot(a, b):.6f}   (lengths {length(a):.4f}, {length(b):.4f})")
for gap in [1, 2, 10, 100]:
    a = rot([the], start=0)[0]; b = rot([cat], start=gap)[0]
    print(f"  gap {gap:<3}: {dot(a, b):.6f}")
