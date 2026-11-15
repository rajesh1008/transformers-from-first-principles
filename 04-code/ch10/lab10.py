"""Lab 10 (second part): three ways to break the variants and the tests that catch them, then the
checkpoint.  Run from 04-code/ch10:  python3 lab10.py"""
from _common import *
from tiny.attention import MultiHead, softmax, causal_mask, window_mask, split_heads
from tiny.embed import Embedding, PositionTable
from tiny.tokenizer import load_bpe

tok = load_bpe(TINY_BPE)
ids = tok.encode(SENTENCE)
Xs = (Embedding(len(tok))(ids) + PositionTable()(range(len(ids)))) * 50   # scaled as in Lab 10, first part
n = len(ids)

# Breakage 1: tile instead of repeat when handing key-value heads to query heads
m = MultiHead(n_heads=4, n_kv_heads=2)
good, A_good = m(Xs, return_shares=True)
Q = split_heads(Xs @ m.W_q, 4); K = split_heads(Xs @ m.W_k, 2); V = split_heads(Xs @ m.W_v, 2)
Kt, Vt = np.tile(K, (2, 1, 1)), np.tile(V, (2, 1, 1))            # wrong: order 0,1,0,1
S = Q @ Kt.transpose(0, 2, 1) / np.sqrt(m.d_head)
A_bad = softmax(np.where(causal_mask(n), S, -np.inf))
print("Breakage 1, tile for repeat: heads get key-value blocks", [0, 1, 0, 1], "instead of", [0, 0, 1, 1])
print(f"  largest change in any share {np.abs(A_bad - A_good).max():.3f}; rows still sum to one: {bool(np.allclose(A_bad.sum(-1), 1))}")
print("  shapes are identical, so no error is raised. The test that catches it: build the model with")
print("  n_kv_heads=2 and check that heads 0 and 1 give the same shares as a one-head model using key block 0:")
one = MultiHead(n_heads=4, n_kv_heads=4)
one.W_k = np.concatenate([m.W_k[:, :32]] * 2 + [m.W_k[:, 32:]] * 2, axis=1)
one.W_v = np.concatenate([m.W_v[:, :32]] * 2 + [m.W_v[:, 32:]] * 2, axis=1)
_, A_ref = one(Xs, return_shares=True)
print(f"  correct code vs reference: {np.abs(A_good - A_ref).max():.1e}; broken code vs reference: {np.abs(A_bad - A_ref).max():.3f}")

# Breakage 2: off-by-one window, strict < where <= belongs
def window_mask_bad(n, w):
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    return (j <= i) & (i - j <= w)                     # lets in w + 1 seats
for name, M in [("correct window_mask(6, 3)", window_mask(6, 3)), ("off by one", window_mask_bad(6, 3))]:
    print(f"\nBreakage 2, {name}: seats each row may see {M.sum(1).tolist()}")
print("  test: no row may see more than `window` seats, and the last row sees exactly `window`.")

# Breakage 3: window mask applied without the causal mask
def window_only(n, w):
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    return np.abs(i - j) < w                           # a band on both sides: the future leaks in
M = window_only(6, 3)
print(f"\nBreakage 3, band without causality: 'the' (seat 0) may see seats {np.where(M[0])[0].tolist()}")
S = (X @ W_q) @ (X @ W_k).T / np.sqrt(2)
A = softmax(np.where(M, S, -np.inf))
print(f"  share 'the' gives to 'cat', a word not yet written: {A[0, 1]:.3f}")
print("  test: change the last token and check that no earlier output moves (Chapter 8's leak test).")
X2 = X.copy(); X2[5] = [9.0, -9.0]
moved = np.abs(softmax(np.where(M, (X2 @ W_q) @ (X2 @ W_k).T / np.sqrt(2), -np.inf))[:5] - A[:5]).max()
print(f"  with the band, earlier rows move by up to {moved:.3f} when only the last word changes")

# Checkpoint
print("\nCheckpoint:")
checks = {
    "defaults equal Chapter 9": np.array_equal(MultiHead(n_heads=4)(Xs), MultiHead(n_heads=4, n_kv_heads=4)(Xs)),
    "grouped heads share tables in the right order": np.allclose(A_good, A_ref),
    "no row sees more than the window": int(window_mask(64, 8).sum(1).max()) == 8,
    "window as long as the text changes nothing": np.allclose(MultiHead(n_heads=4)(Xs), MultiHead(n_heads=4, window=n)(Xs)),
    "nothing leaks from the future with a window": np.allclose(
        MultiHead(n_heads=4, window=3)(Xs)[:-1],
        MultiHead(n_heads=4, window=3)(np.vstack([Xs[:-1], Xs[:1] * 3]))[:-1]),
    "cache per token for 2 kv heads of 32 is 128": 2 * 2 * 32 == 128,
}
for k, v in checks.items():
    print(f"  [{'pass' if v else 'FAIL'}] {k}")
print("all pass:", all(checks.values()))

# The lab's last table: Tiny, four query heads, two key-value heads, window 3 (figure 10.8)
_, A = MultiHead(n_heads=4, n_kv_heads=2, window=3)(Xs, return_shares=True)
print(f"\nTiny, 4 query heads, 2 kv heads, window 3, {n} tokens; last row of each head:")
for h in range(4):
    print(f"  head {h} (kv block {h // 2}): {r3(A[h, -1])}")
print("heads 0 and 1 share keys but differ:", not np.allclose(A[0], A[1]))
