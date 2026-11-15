"""Chapter 10, section 5: mixing local and global attention.  Two ways to give a windowed model a
long view: (1) global seats that every token may see and that see every token (the Longformer idea),
(2) interleaving full-attention layers among window layers.  Running sentence with Chapter 8's hand head,
then memory for GPT-2 small's shape.  Run from 04-code/ch10:  python3 local_global.py"""
from _common import *
from tiny.attention import softmax, causal_mask, window_mask

def global_mask(n, window, globals_):
    """Window mask plus global seats: a global seat may be seen by every later token, and a global
    token may look at every earlier seat.  Never the future."""
    M = window_mask(n, window).copy()
    C = causal_mask(n)
    for g in globals_:
        M[:, g] |= C[:, g]          # everyone after g may look at g
        M[g, :] |= C[g, :]          # g may look at everything before it
    return M

Q, K, V = X @ W_q, X @ W_k, X @ W_v
S = Q @ K.T / np.sqrt(2)
full = softmax(np.where(causal_mask(6), S, -np.inf))
for name, M in [("window 3", window_mask(6, 3)),
                ("window 3 + global seat 0 (the first 'the')", global_mask(6, 3, [0])),
                ("window 3 + global seat 1 (cat)", global_mask(6, 3, [1]))]:
    A = softmax(np.where(M, S, -np.inf)); out = A @ V
    print(f"{name}:\n{M.astype(int)}")
    print(f"  mat's row {r3(A[5])} | share to cat {A[5, 1]:.3f} | mat's output {r3(out[5])}"
          f" | allowed pairs {int(M.sum())} of {int(causal_mask(6).sum())}\n")
print(f"no window: mat's row {r3(full[5])}, output {r3((full @ V)[5])}")

# allowed pairs grow linearly with length for window + g global seats
print("\nallowed (query, key) pairs, causal, as the text grows (window 256, 1 global seat):")
for n in (1024, 8192, 65536):
    w, g = 256, 1
    full_pairs = n * (n + 1) // 2
    win_pairs = sum(min(i + 1, w) for i in range(n))
    glob_extra = (n - w) if n > w else 0          # rows further than the window back from seat 0 gain it
    print(f"  n={n:>6,}: full {full_pairs:>13,} | window {win_pairs:>11,} | window + global {win_pairs + glob_extra:>11,}"
          f" | ratio full/window {full_pairs / win_pairs:6.1f}")

# interleaving: what a stack can reach and what it must remember
print("\nGPT-2 small's shape (12 layers, 12 kv heads of 64, 16-bit), cache at 65,536 tokens:")
per_tok_layer = 12 * 64 * 2 * 2                      # bytes per token per layer: heads x d_head x (k,v) x 2 bytes
for name, n_full in [("all 12 layers full", 12), ("1 full layer in 4 (3 full, 9 window 256)", 3),
                     ("1 full layer in 6 (2 full, 10 window 256)", 2), ("all 12 layers window 256", 0)]:
    n_win = 12 - n_full
    b = n_full * 65536 * per_tok_layer + n_win * 256 * per_tok_layer
    print(f"  {name:<44} {b / 2**20:8.1f} MiB")
print("reach of the last token of 65,536 with a window of 256: 12 window layers ->",
      max(0, 65535 - 12 * 255), "seats back is the earliest; any full layer -> seat 0")
print(f"  of the 1-in-4 mix: 3 full layers {3 * 65536 * per_tok_layer / 2**20:.1f} MiB, 9 window layers {9 * 256 * per_tok_layer / 2**20:.2f} MiB;"
      f" one full layer alone {65536 * per_tok_layer / 2**20:.1f} MiB")
gq = 4 * 64 * 2 * 2
print(f"  1-in-4 mix with grouped-query (4 kv heads) in every layer: {(3 * 65536 + 9 * 256) * gq / 2**20:.1f} MiB")
