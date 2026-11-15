"""Chapter 9: two heads by hand on the running sentence, each of width 1, cut from Chapter 8's
hand tables.  Chapter 8's head added two products in every score (width 2).  Cut its query, key and
value tables into their first and second columns and you have two heads of width 1: head 1 keeps
the first column of each table, head 2 the second.  Each scores, masks and softmaxes on its own,
scaled by the square root of 1.  Then the two outputs are put side by side and mixed by an output
table.  Run from 04-code/ch09:  python3 hand_two_heads.py"""
from _common import *
from tiny.attention import attention, softmax, causal_mask

Q, K, V = X @ W_q, X @ W_k, X @ W_v
print("queries (Chapter 8):", Q.astype(int).tolist())
print("keys    (Chapter 8):", K.astype(int).tolist())
print("values  (Chapter 8):", V.astype(int).tolist())
one, A1 = attention(X, W_q, W_k, W_v)
outs, tabs = [], []
for h in (0, 1):
    q, k, v = Q[:, h:h + 1], K[:, h:h + 1], V[:, h:h + 1]
    S = q @ k.T                                   # width 1: each score is one product
    A = softmax(np.where(causal_mask(6), S / np.sqrt(1), -np.inf))
    outs.append(A @ v); tabs.append(A)
    print(f"\nhead {h + 1} (column {h + 1} of each table): raw scores")
    print(table(S, fmt="{:>7.0f}"))
    print(f"head {h + 1} Attention Table")
    print(table(A))
raw_sum = (Q[:, :1] @ K[:, :1].T) + (Q[:, 1:] @ K[:, 1:].T)
print("\nthe two heads' raw score tables add up to Chapter 8's raw table:", bool(np.allclose(raw_sum, Q @ K.T)))
print("sat's row, one head of width 2 :", r3(A1[2, :3]))
print("sat's row, head 1 of width 1   :", r3(tabs[0][2, :3]))
print("sat's row, head 2 of width 1   :", r3(tabs[1][2, :3]))
cat = np.concatenate(outs, axis=1)
print("\nthe two outputs side by side (6 x 2):")
for w, row in zip(WORDS, cat):
    print(f"  {w:>4} {r3(row)}")
print("one head of width 2, for comparison:")
for w, row in zip(WORDS, one):
    print(f"  {w:>4} {r3(row)}")
W_o = np.array([[1, 0], [0, 1]], dtype=float)
print("\nwith the output table [[1, 0], [0, 1]] (keep each head's number in its own place), sat's output:", r3((cat @ W_o)[2]))
W_o2 = np.array([[1, 1], [0, 1]], dtype=float)
print("with the output table [[1, 1], [0, 1]] (head 1 also writes into the second place), sat's output:", r3((cat @ W_o2)[2]))
