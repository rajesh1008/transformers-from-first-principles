"""Chapter 9: giving Head the option of Chapter 6's Rotary.  With rotary=True each head turns its
queries and keys (never its values) by clocks sized to its own width, d_head = 32.  The same
sentence placed at seat 0 and at seat 100 then gives the same Attention Table; with Chapter 5's
learned positions added to the rows it does not.  Run from 04-code/ch09:  python3 rotary_heads.py"""
from _common import *
from tiny.attention import Head, MultiHead
from tiny.embed import Embedding, PositionTable, Rotary
from tiny.tokenizer import load_bpe

tok = load_bpe(TINY_BPE)
ids = tok.encode(SENTENCE)
emb, pos = Embedding(len(tok)), PositionTable()
T = emb(ids) * 50                                   # token rows only, scaled up so that untrained shares are not all even
h = Head(d_head=32, rotary=True)
_, A0 = h(T, return_shares=True, start=0)
_, A100 = h(T, return_shares=True, start=100)
print("Head(d_head=32, rotary=True): learned numbers", f"{h.n_params:,}", "(rotary adds none); clocks per head:", len(h.rotary.speeds))
print("last row at seat 0  :", r3(A0[-1]))
print("last row at seat 100:", r3(A100[-1]))
print("same Attention Table at seat 0 and seat 100:", bool(np.allclose(A0, A100)))
P0 = T + pos(range(len(ids))) * 50; P100 = T + pos(range(100, 100 + len(ids))) * 50
hp = Head(d_head=32)
_, B0 = hp(P0, return_shares=True); _, B100 = hp(P100, return_shares=True)
print("learned positions added instead: same table at seat 0 and 100:", bool(np.allclose(B0, B100)),
      f" (largest change in a share: {np.abs(B0 - B100).max():.3f})")
m = MultiHead(n_heads=4, rotary=True)
_, M0 = m(T, return_shares=True, start=0); _, M100 = m(T, return_shares=True, start=100)
print("MultiHead(n_heads=4, rotary=True): every head's table unchanged by the move:", bool(np.allclose(M0, M100)))
print("values untouched by rotary: head output rows identical too:", bool(np.allclose(m(T, start=0), m(T, start=100))))
r = Rotary(d=32)
print("clock speeds of a 32-wide head (fastest, slowest), turns per seat in radians:", f"{r.speeds[0]:.4f}", f"{r.speeds[-1]:.6f}")
r128 = Rotary(d=128)
print("for comparison a 128-wide head's slowest clock:", f"{r128.speeds[-1]:.6f}")
