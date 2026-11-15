"""Section 5.5: learned and sinusoidal positions on the running sentence.  Run: python3 compare.py"""
import _common
from _common import SENTENCE, TINY_BPE
import numpy as np
from tiny.tokenizer import TOY_VOCAB, load_bpe, TINY_VOCAB_SIZE
from tiny.embed import (TOY_EMBED, TOY_POS, clock, add_positions, cosine, length, dot,
                        Embedding, PositionTable, sinusoidal, embed_with_positions)

words = SENTENCE.split()
X = TOY_EMBED[[TOY_VOCAB.index(w) for w in words]]
L = add_positions(X, TOY_POS)
CLK = np.array([clock(k) for k in range(6)])
S = add_positions(X, CLK)
print(f"{'':<14}{'learned (hand-chosen)':<24}{'clock (30 deg per token)':<24}")
for k, w in enumerate(words):
    print(f"  {k} {w:<4}      ({L[k][0]:>5.2f}, {L[k][1]:>5.2f})          ({S[k][0]:>5.2f}, {S[k][1]:>5.2f})")
print("cosine of the two 'the's: learned", round(cosine(L[0], L[4]), 3), "; clock", round(cosine(S[0], S[4]), 3), "; no positions 1.0")
print("agreement of each position row with the next one (gap 1): learned",
      [float(dot(TOY_POS[k], TOY_POS[k + 1])) for k in range(5)], "; clock", [round(float(dot(CLK[k], CLK[k + 1])), 3) for k in range(5)])
print("how much of each sum is word and how much is position (lengths):")
print("  word rows  ", [round(float(length(r)), 2) for r in X])
print("  learned pos", [round(float(length(r)), 2) for r in TOY_POS])
print("  clock pos  ", [round(float(length(r)), 2) for r in CLK])

tok = load_bpe(TINY_BPE)
ids = tok.encode(SENTENCE)
emb = Embedding(TINY_VOCAB_SIZE, seed=1865)
pos = PositionTable(seed=1865)
SIN = sinusoidal(1024)
A = embed_with_positions(ids, emb, pos)
B = embed_with_positions(ids, emb, SIN)
tl = np.median([length(r) for r in emb(ids)])
print(f"Tiny, before training: token rows ~{tl:.3f} long; learned position rows ~{np.median([length(r) for r in pos(range(7))]):.3f}; "
      f"sinusoidal rows {length(SIN[0]):.3f}")
print(f"  share of the sum's direction that is the word (cosine with the token row): learned "
      f"{np.mean([cosine(a, t) for a, t in zip(A, emb(ids))]):.3f}; sinusoidal {np.mean([cosine(b, t) for b, t in zip(B, emb(ids))]):.3f}")
scaled = emb.table * 128 ** 0.5
print(f"  with token rows multiplied by the square root of 128 (as the 2017 Transformer did): token rows ~{np.median([length(r) for r in scaled[ids]]):.2f}, "
      f"cosine with the token row {np.mean([cosine(scaled[i] + SIN[k], scaled[i]) for k, i in enumerate(ids)]):.3f}")
