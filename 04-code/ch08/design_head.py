"""Chapter 8: design a head by hand so that sat finds cat, and see what the same tables do
to every other row.  Run from 04-code/ch08:  python3 design_head.py"""
from _common import *
from tiny.attention import attention, entropy

W_k_id = np.eye(2)                               # keys: each row unchanged
print("sat's row is (1, 1); cat's is (0, 1).  Keys = rows.  Try queries that turn (1, 1) into (-c, c).")
for c in (1, 2, 3, 5):
    W_q_d = np.array([[-c, 0], [0, c]], dtype=float)
    out, A = attention(X, W_q_d, W_k_id, W_v)
    print(f"  c = {c}: sat's query {tuple(int(v) for v in X[2] @ W_q_d)}  sat's row {np.round(A[2, :3], 3).tolist()}  "
          f"sat's output {tuple(round(float(v), 3) for v in out[2])}")
W_q_d = np.array([[-3, 0], [0, 3]], dtype=float)
out, A = attention(X, W_q_d, W_k_id, W_v)
print("\nthe whole Attention Table with c = 3 (the same two tables serve every token)")
print(table(A))
print("  entropy per row:", np.round(entropy(A), 3).tolist())
print("  largest share in each row goes to:", [WORDS[j] for j in A.argmax(axis=1)])
