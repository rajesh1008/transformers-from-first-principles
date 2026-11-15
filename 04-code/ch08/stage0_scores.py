"""Stage 0 (Voice Reference): the 6x6 score table for one attention head, by hand and in one line.
Run from 04-code/ch08:  python3 stage0_scores.py"""
import numpy as np

words = ["the", "cat", "sat", "on", "the", "mat"]
X = np.array([[1, 0], [0, 1], [1, 1], [1, -1], [1, 0], [-1, 1]])   # six 2-D embeddings, chosen by hand
W_q = np.array([[1, 0], [0, 2]])        # the question each word asks
W_k = np.array([[0, 1], [1, 0]])        # the label each word wears
Q, K = X @ W_q, X @ W_k
scores = Q @ K.T                        # all thirty-six agreements in one line
print("queries:", Q.tolist())
print("keys:   ", K.tolist())
print("scores (rows ask, columns answer):")
print("        " + "".join(f"{w:>6}" for w in words))
for w, row in zip(words, scores):
    print(f"{w:>6}  " + "".join(f"{v:>6}" for v in row))
