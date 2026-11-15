"""Chapter 8: what a head knows about order.  (1) Without positions, shuffling the earlier words
leaves the last token's output unchanged.  (2) Learned positions added to the rows fix it.
(3) Rotary positions, applied to queries and keys only (Chapter 6), make scores depend on gaps.
Run from 04-code/ch08:  python3 order.py"""
from _common import *
from tiny.attention import attention, softmax, causal_mask
from tiny.embed import TOY_POS, TOY_TURN, apply_rotary

perm = [1, 0, 2, 3, 4, 5]                            # 'cat the sat on the mat'
Xs = X[perm]
for name, A_, B_ in (("no positions", X, Xs), ("toy positions added", X + TOY_POS, Xs + TOY_POS)):
    a, _ = attention(A_, W_q, W_k, W_v); b, _ = attention(B_, W_q, W_k, W_v)
    print(f"{name:<22} 'the cat ...' mat -> {tuple(np.round(a[5], 3).tolist())}   'cat the ...' mat -> {tuple(np.round(b[5], 3).tolist())}"
          f"   same: {bool(np.allclose(a[5], b[5]))}")

def rotary_head(X_, start=0):
    pos = list(range(start, start + len(X_)))
    Q = apply_rotary(X_ @ W_q, pos, speeds=TOY_TURN); K = apply_rotary(X_ @ W_k, pos, speeds=TOY_TURN)
    S = np.where(causal_mask(len(X_)), Q @ K.T / np.sqrt(2), -np.inf)
    A = softmax(S); return A @ (X_ @ W_v), A
a, A = rotary_head(X); b, _ = rotary_head(Xs)
print(f"\nrotary on queries and keys (30 degrees per seat): mat -> {tuple(np.round(a[5], 3).tolist())} vs shuffled {tuple(np.round(b[5], 3).tolist())}"
      f"   same: {bool(np.allclose(a[5], b[5]))}")
print("  sat's row:", np.round(A[2, :3], 3).tolist(), " (no positions: [0.284, 0.14, 0.576])")
a100, A100 = rotary_head(X, start=100)
print("  the same sentence starting at seat 100: tables identical:", bool(np.allclose(A, A100)), " outputs identical:", bool(np.allclose(a, a100)))
