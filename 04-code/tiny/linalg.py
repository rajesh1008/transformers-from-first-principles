"""Tiny's matrix helpers (Chapter 7).  A matrix is a stack of rows; `A @ B` makes every
row of A meet every column of B in a dot product.  These few lines replace the loops of
Chapter 6's `agreement_table` (kept in tiny/embed.py as the reference) everywhere from Chapter 8 on."""
import numpy as np


def agreement_table_fast(asking, answering=None):
    """Every asking row's dot product with every answering row, in one line.
    Shapes: (n, d) @ (d, m) -> (n, m).  Row = the token asking, column = the token answering.
    Same numbers as tiny.embed.agreement_table, computed as one matrix multiplication."""
    asking = np.asarray(asking, dtype=float)
    answering = asking if answering is None else np.asarray(answering, dtype=float)
    return asking @ answering.T


def matmul_loops(A, B):
    """Matrix multiplication with three plain loops, exactly as it is defined:
    entry (i, j) = sum over k of A[i, k] * B[k, j].  For checking and for timing only."""
    n, k = len(A), len(A[0])
    k2, m = len(B), len(B[0])
    if k != k2:
        raise ValueError(f"inner sizes differ: ({n}, {k}) @ ({k2}, {m})")
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for t in range(k):
                s += A[i][t] * B[t][j]
            out[i][j] = s
    return out


def matmul_tiled(A, B, tile=128):
    """The same product, computed one tile of the answer at a time.  Each tile needs only a
    strip of A and a strip of B, which is how fast chips keep the working numbers close at hand."""
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    n, k = A.shape
    m = B.shape[1]
    out = np.zeros((n, m))
    for i in range(0, n, tile):
        for j in range(0, m, tile):
            for t in range(0, k, tile):
                out[i:i + tile, j:j + tile] += A[i:i + tile, t:t + tile] @ B[t:t + tile, j:j + tile]
    return out


def work(n, k, m):
    """Arithmetic in an (n, k) @ (k, m) product: n*m entries, each k multiplications and k-1 additions."""
    return n * m * k, n * m * (k - 1)


def numbers_moved(n, k, m):
    """Numbers that must be read (both inputs) and written (the answer) at least once."""
    return n * k + k * m + n * m


def intensity(n, k, m, bytes_per_number=2):
    """Arithmetic intensity: operations (multiplications + additions, about 2nkm) per byte moved."""
    return 2 * n * k * m / (bytes_per_number * numbers_moved(n, k, m))


def check_shapes(a_shape, b_shape):
    """The shape rule for a @ b: inner sizes must match; the answer takes the outer sizes."""
    (n, k), (k2, m) = a_shape, b_shape
    if k != k2:
        raise ValueError(f"cannot multiply {a_shape} by {b_shape}: inner sizes {k} and {k2} differ")
    return (n, m)
