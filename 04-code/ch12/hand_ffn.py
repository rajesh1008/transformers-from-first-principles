"""Chapter 12, sections 1 and 2: a feed-forward block small enough to do by hand.  Three neurons on the
2-number toy rows.  Input: each word's row after Chapter 8's hand head, with the head's output added
back to the row (the residual, which Chapter 13 explains).  Run from 04-code/ch12:  python3 hand_ffn.py"""
from _common import *
from tiny.attention import attention
from tiny.ffn import relu

out8, _ = attention(X, W_q, W_k, W_v)
R = X + out8                                     # each row plus what attention brought it
W_1 = np.array([[1, 0, 1],                       # columns are the keys: k1 = (1, 0), k2 = (0, 1), k3 = (1, 1)
                [0, 1, 1]], dtype=float)
b_1 = np.array([0, 0, -2], dtype=float)          # neuron 3 needs its match to exceed 2 before it fires
W_2 = np.array([[0, 0.5],                        # rows are the values: v1 = (0, 0.5), v2 = (0.5, 0), v3 = (-0.5, 0.5)
                [0.5, 0],
                [-0.5, 0.5]], dtype=float)

print("Hand feed-forward block: 2 -> 3 -> 2, ReLU")
print("  keys (columns of W_1): k1 = (1, 0), k2 = (0, 1), k3 = (1, 1); biases b_1 = (0, 0, -2)")
print("  values (rows of W_2):  v1 = (0, 0.5), v2 = (0.5, 0), v3 = (-0.5, 0.5); no output bias")
print("\nrow after attention (x + head output), match with each key (x W_1 + b_1), activation, block output, new row:")
up = R @ W_1 + b_1
h = relu(up)
F = h @ W_2
for w, r, u, a, f in zip(WORDS, R, up, h, F):
    print(f"  {w:>3}: row {r3(r)} | matches {r3(u)} | active {r3(a)} | adds {r3(f)} | new row {r3(r + f)}")
print("neurons active per word:", [int((a > 0).sum()) for a in h], "| words that fire neuron 3:",
      [WORDS[i] for i in range(6) if h[i, 2] > 0])

perm = [5, 2, 0, 4, 1, 3]
print("\nrow by row: shuffling the rows shuffles the outputs and changes nothing else:",
      bool(np.allclose((relu(R[perm] @ W_1 + b_1) @ W_2), F[perm])))
print("mat's output from mat's row alone, with no other word present:", r3(relu(R[5:6] @ W_1 + b_1) @ W_2))

W12 = W_1 @ W_2
print("\nwithout the activation the two tables collapse into one 2 x 2 table W_1 W_2 =", W12.tolist())
lin = R @ W_1 @ W_2 + b_1 @ W_2
print("  block output without ReLU equals x (W_1 W_2) + b_1 W_2 for every row:", bool(np.allclose((R @ W_1 + b_1) @ W_2, lin)))
grid = np.array([[a, b] for a in (-1.0, 0.0, 1.0, 2.0) for b in (-1.0, 0.0, 1.0, 2.0)])
g3 = relu(grid @ W_1[:, 2] + b_1[2])
print("  neuron 3 on a grid of rows (a, b) in {-1, 0, 1, 2}^2 fires on:", [tuple(int(v) for v in p) for p, a in zip(grid, g3) if a > 0])
best = np.linalg.lstsq(np.c_[grid, np.ones(len(grid))], g3, rcond=None)
res = g3 - np.c_[grid, np.ones(len(grid))] @ best[0]
print(f"  best straight-line fit to neuron 3 over the grid leaves a largest error of {np.abs(res).max():.3f}: no single table can do it")
