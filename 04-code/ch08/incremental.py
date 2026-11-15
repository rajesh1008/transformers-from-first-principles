"""Chapter 8: the head while writing.  With the causal mask, adding a token never changes the
earlier rows, and the new token's row needs only its own query against all keys so far.
Run from 04-code/ch08:  python3 incremental.py"""
from _common import *
from tiny.attention import attention, softmax

out5, A5 = attention(X[:5], W_q, W_k, W_v)           # 'the cat sat on the'
out6, A6 = attention(X, W_q, W_k, W_v)               # '... mat'
print("outputs for the first five tokens, before and after 'mat' arrives, identical:", bool(np.allclose(out5, out6[:5])))
q = X[5] @ W_q; K = X @ W_k; V = X @ W_v             # only mat's query is new
row = softmax(q @ K.T / np.sqrt(2))
print("mat's row from its query alone:", np.round(row, 3).tolist())
print("mat's row from the full table: ", np.round(A6[5], 3).tolist())
print("mat's output, one row:", tuple(np.round(row @ V, 3).tolist()), " full table:", tuple(np.round(out6[5], 3).tolist()))
n = 1024
print(f"\nwork for the newest token of a {n:,}-token window: {n:,} scores (one row); whole table: {n * n:,}")
print(f"keys and values kept from earlier tokens: 2 x {n - 1:,} rows; the earlier queries are never needed again")
out_open5, _ = attention(X[:5], W_q, W_k, W_v, causal=False); out_open6, _ = attention(X, W_q, W_k, W_v, causal=False)
print("without the mask, the earlier outputs change when 'mat' arrives:", not np.allclose(out_open5, out_open6[:5]))
