"""Chapter 10, sections 4 and lab: sliding windows.  Each token may look at itself and the window - 1
tokens before it.  On the running sentence with Chapter 8's hand head, a window of 3 lets mat see
on, the and itself, and nothing of cat.  Then two layers of windows: what a second layer can reach.
Run from 04-code/ch10:  python3 window.py"""
from _common import *
from tiny.attention import attention, softmax, causal_mask, window_mask, MultiHead

print("window mask, width 3 (1 = may look):")
print(window_mask(6, 3).astype(int))
Q, K, V = X @ W_q, X @ W_k, X @ W_v
S = Q @ K.T / np.sqrt(2)
for w in (6, 4, 3, 2):
    A = softmax(np.where(window_mask(6, w), S, -np.inf))
    out = A @ V
    print(f"\nwindow {w}: mat's row {r3(A[5])}  share to cat {A[5, 1]:.3f} | mat's output {r3(out[5])}"
          f" | sat's row {r3(A[2, :3])}")
full = softmax(np.where(causal_mask(6), S, -np.inf)) @ V
A3 = softmax(np.where(window_mask(6, 3), S, -np.inf)); w3 = A3 @ V
changed = [WORDS[i] + str(i) for i in range(6) if not np.allclose(full[i], w3[i])]
print("\noutputs changed by a window of 3 (against no window):", changed)

# two layers of windows: what can a token reach?
def reach(n, w, layers):
    M = window_mask(n, w).astype(int)
    R = np.eye(n, dtype=int)
    for _ in range(layers):
        R = (M @ R > 0).astype(int)
    return R
for L in (1, 2, 3):
    r = reach(6, 3, L)
    print(f"window 3, {L} layer{'s' if L > 1 else ''}: mat can be influenced by {[WORDS[j] + str(j) for j in range(6) if r[5, j]]}")
n, w = 1024, 256
for L in (1, 2, 4, 12):
    print(f"window {w}, {L:>2} layers: the last of {n} tokens can draw on seats back to {max(0, n - 1 - L * (w - 1))}")

# Tiny's layer with a window: the shares outside the window are exactly zero, the rest renormalised
from tiny.embed import Embedding, PositionTable
from tiny.tokenizer import load_bpe
tok = load_bpe(TINY_BPE); ids = tok.encode(SENTENCE)
Xt = Embedding(len(tok))(ids) * 50 + PositionTable()(range(len(ids))) * 50
m_full, m_win = MultiHead(n_heads=4), MultiHead(n_heads=4, window=3)
_, Af = m_full(Xt, return_shares=True); _, Aw = m_win(Xt, return_shares=True)
print("\nTiny, four heads, window 3, last row of head 0:", r3(Aw[0, -1]), "| without window:", r3(Af[0, -1]))
print("rows of shares still sum to one:", bool(np.allclose(Aw.sum(-1), 1)))
