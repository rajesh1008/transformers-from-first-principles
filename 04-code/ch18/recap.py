"""Chapter 18, section 18.1: the forward pass of layer 0's head in the hand Tiny, every intermediate named.
The loss reads five seats (the@0 .. the@4 predict cat .. mat); mat@5 is only a target, so the head's tables
here are 5 rows long.  64-bit; nothing random."""
import _common
import numpy as np
from hand import hand_tiny, IDS
from tiny.attention import head_forward
from _common import SEATS, save_json, f4

m = hand_tiny()
b = m.blocks[0]
X = m.embed(IDS[:-1])                                   # token row + position row, seats 0..4
A = b.ln_1(X)
f = head_forward(A, b.att.W_q, b.att.W_k, b.att.W_v, b.att.W_o)
def show(name, T, k=4):
    print(f"{name}:")
    for s, r in enumerate(np.atleast_2d(T)):
        cells = "  ".join(f4(v, k) if np.isfinite(v) else "   -inf" for v in r)
        print(f"  {SEATS[s]:<6} {cells}")
print("hand Tiny, layer 0's head, on the five seats the loss reads (Chapter 15's tables)")
show("X, the rows entering layer 0 (token row + position row)", X, 3)
show("A = RMSNorm(X), the rows the head reads", A)
show("Q = A W_q", f["Q"]); show("K = A W_k", f["K"]); show("V = A W_v", f["V"])
show("S = Q K^T / sqrt(2), future masked", f["S"])
show("P = softmax(S), row by row", f["P"])
show("O = P V, the blend", f["O"])
print("Y = O W_o: W_o is the identity, so Y = O")
print(f"loss of the whole hand Tiny on the sentence: {m.loss(IDS):.4f}")
save_json("recap", {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in f.items() if k != "W_o" or v is not None} | {"X": X.tolist()})
