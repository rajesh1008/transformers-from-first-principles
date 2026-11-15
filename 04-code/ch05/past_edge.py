"""Section 5.6: what happens past the longest position.  Run: python3 past_edge.py"""
import _common
from _common import TINY_BPE
import numpy as np
import _gpt2
from tiny.corpus import load_text
from tiny.tokenizer import TOY_VOCAB, load_bpe
from tiny.embed import (PositionTable, sinusoidal, TOY_EMBED, TOY_POS, add_positions, cosine, length,
                        TINY_CONTEXT)

pos = PositionTable(seed=1865)
print("Tiny's learned table has rows for positions 0 to", pos.context - 1)
for p in (1023, 1024):
    try:
        row = pos([p])
        print(f"  position {p}: a row, starting {np.round(row[0][:3], 4).tolist()}")
    except IndexError as err:
        print(f"  position {p}: IndexError: {err}")

P = _gpt2.tensor("wpe.weight")
print(f"GPT-2: last row (position 1023) has length {length(P[1023]):.3f}; position 1022 has {length(P[1022]):.3f}; "
      f"cosine between them {cosine(P[1022], P[1023]):.3f}")
try:
    P[1024]
except IndexError as err:
    print("GPT-2 position 1024: IndexError:", err)

# The fix every system uses first: cut the text into windows that fit
ids = load_bpe(TINY_BPE).encode(load_text())
full, rest = divmod(len(ids), TINY_CONTEXT)
print(f"the novel is {len(ids):,} tokens: {full} full windows of {TINY_CONTEXT:,} and one of {rest}")

# Sinusoidal rows exist for any position. Are they like any row the model trained on?
S = sinusoidal(8192)
U = S / 8.0
trained = U[:TINY_CONTEXT]
print("sinusoidal: closest trained row (0-1023) for positions past the end:")
for p in (1024, 1100, 1536, 2048, 4096, 8191):
    c = trained @ U[p]
    print(f"  position {p:>4}: nearest trained position {int(c.argmax()):>4}, cosine {c.max():.3f}")
inside = trained @ trained.T
np.fill_diagonal(inside, -2)
print(f"  (inside the range, every row's nearest other row has cosine at least {inside.max(axis=1).min():.3f})")
far = [float((trained @ U[p]).max()) for p in range(2048, 8192)]
print(f"  positions 2048-8191: nearest-trained cosine averages {np.mean(far):.3f}, lowest {min(far):.3f}")

# Clamping: give every position past the end the last row
def clamp_rows(n, table):
    return np.array([table[min(k, len(table) - 1)] for k in range(n)])
words = "the cat sat on the mat and the dog".split()
known = [w if w in TOY_VOCAB else "<unk>" for w in words]
X = TOY_EMBED[[TOY_VOCAB.index(w) for w in known]]
Y = add_positions(X, clamp_rows(len(words), TOY_POS))
print("toy table has 6 rows; a 9-token sentence with clamping gives positions", [min(k, 5) for k in range(9)])
print("  tokens 5-8 all received position row", TOY_POS[5].tolist(), "-> the order of the last four is invisible again")
