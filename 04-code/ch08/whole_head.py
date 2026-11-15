"""Chapter 8: the whole head in twelve lines, with nothing imported but NumPy.
Run from 04-code/ch08:  python3 whole_head.py"""
import numpy as np
X = np.array([[1, 0], [0, 1], [1, 1], [1, -1], [1, 0], [-1, 1]], dtype=float)  # 1 six rows
W_q = np.array([[1, 0], [0, 2]]); W_k = np.array([[0, 1], [1, 0]])              # 2 asking, labelling
W_v = np.array([[1, 0], [1, 1]])                                                # 3 offering
Q, K, V = X @ W_q, X @ W_k, X @ W_v                                             # 4 three rows each
scores = Q @ K.T / np.sqrt(K.shape[1])                                          # 5 agreements, scaled
n = len(X)                                                                      # 6
mask = np.tril(np.ones((n, n), dtype=bool))                                     # 7 no peeking
scores = np.where(mask, scores, -np.inf)                                        # 8
e = np.exp(scores - scores.max(axis=1, keepdims=True))                          # 9 exponentials
shares = e / e.sum(axis=1, keepdims=True)                                       # 10 shares sum to one
out = shares @ V                                                                # 11 weighted sums
print(np.round(shares, 3)); print(np.round(out, 3))                             # 12
