"""Lab 11: tiled attention by hand and in code, three breakages and a checkpoint.
Part 1 recovers the running sentence's 6 x 6 share table from two 3-wide tiles, using only each row's final
running maximum and running sum.  Part 2 breaks the running quantities three ways, each of which runs without
an error.  Part 3 is the checkpoint: five tests, each of which catches one mistake.
Run from 04-code/ch11:  python3 lab11.py"""
from _common import *
from tiny.attention import attention, tiled_attention, softmax, causal_mask

Q, K, V = X @ W_q, X @ W_k, X @ W_v
d = Q.shape[1]
out_ref, P_ref = attention(X, W_q, W_k, W_v)            # Chapter 8's hand head, causal

def tiles(Q, K, V, block, causal=True, mistake=None, shift=None, dtype=float):
    """The tiled forward pass written out, with optional mistakes.  Returns output, final m and l per row."""
    n = len(Q); out = np.empty((n, V.shape[1])); M = np.empty(n); L = np.empty(n)
    for i0 in range(0, n, block):
        q = Q[i0:i0 + block] / np.sqrt(d); rows = np.arange(i0, i0 + len(q))
        m = np.full(len(q), -np.inf); l = np.zeros(len(q)); o = np.zeros((len(q), V.shape[1]))
        for j0 in range(0, n, block):
            cols = np.arange(j0, min(j0 + block, n))
            if causal and cols[0] > rows[-1]:
                break
            s = q @ K[j0:j0 + block].T
            if causal:
                s = np.where(cols[None, :] <= rows[:, None], s, -np.inf)
            if shift is not None:                                    # Breakage 3: one fixed shift, no running max
                p = np.exp((s - shift).astype(dtype)).astype(float)
                l = l + p.sum(1); o = o + p @ V[j0:j0 + block]; m = np.full(len(q), float(shift)); continue
            m_new = np.maximum(m, s.max(1))
            m_use = np.where(np.isinf(m_new), 0.0, m_new)
            scale = np.exp(m - m_use)
            if mistake == "no rescale":                              # Breakage 1: the max moves, nothing is rescaled
                scale = np.where(np.isinf(m), 0.0, 1.0)
            p = np.exp(s - m_use[:, None])
            l = l * scale + p.sum(1)
            o = o * (np.where(np.isinf(m), 0.0, 1.0)[:, None] if mistake == "sum only" else scale[:, None]) + p @ V[j0:j0 + block]
            m = m_new
        out[i0:i0 + len(q)] = o / l[:, None]; M[i0:i0 + len(q)] = m; L[i0:i0 + len(q)] = l
    return out, M, L

print("Part 1. The 6 x 6 share table from two 3-wide tiles (Chapter 8's hand head, causal)")
out, M, L = tiles(Q, K, V, block=3)
S = Q @ K.T / np.sqrt(d)
for i, w in enumerate(WORDS):
    print(f"  row {i} ({w:>3}): final running max {M[i]:6.3f} | final running sum {L[i]:.4f}")
P = np.where(causal_mask(6), np.exp(S - M[:, None]) / L[:, None], 0.0)
print("recovered shares, exp(score - max) / sum, row by row:")
print(table(P))
print("equal to Chapter 8's table:", bool(np.allclose(P, P_ref)), f"| largest difference {np.abs(P - P_ref).max():.1e}")
print("output rows equal attention():", bool(np.allclose(out, out_ref)), "| sat's output", r3(out[2]), "| mat's output", r3(out[5]))
print("numbers kept per row to rebuild any share later: 2 (max and sum); the table itself: 21 shares never stored")

print("\nPart 2. Three breakages that run without an error")
b1, _, _ = tiles(Q, K, V, block=3, mistake="no rescale")
b2, _, _ = tiles(Q, K, V, block=3, mistake="sum only")
print("Breakage 1, the max moves and nothing is rescaled:")
for i in (3, 4, 5):
    print(f"  {WORDS[i]:>3}: correct {r3(out_ref[i])} | broken {r3(b1[i])}")
print(f"  largest error {np.abs(b1 - out_ref).max():.3f}; rows whose max grew in their second tile: "
      f"{[WORDS[i] for i in range(3, 6) if S[i, 3:i + 1].max() > S[i, :3].max()]}")
print("Breakage 2, the sum is rescaled and the output is not:")
for i in (3, 4, 5):
    print(f"  {WORDS[i]:>3}: correct {r3(out_ref[i])} | broken {r3(b2[i])}")
print(f"  largest error {np.abs(b2 - out_ref).max():.3f}")
one1, _, _ = tiles(Q, K, V, block=6, mistake="no rescale"); one2, _, _ = tiles(Q, K, V, block=6, mistake="sum only")
print(f"  with one block of 6 both broken versions are exact: {bool(np.allclose(one1, out_ref) and np.allclose(one2, out_ref))}")

rng = np.random.default_rng(SEED)
n, dd = 256, 64
Qr, Kr, Vr = (rng.normal(size=(n, dd)) for _ in range(3))
ramp = np.linspace(0, 16, n)                                       # scores that rise along the row
Kb = Kr + 0.0; Qb = Qr + 0.0
def biased_ref(Qx, Kx, Vx):
    s = Qx @ Kx.T / np.sqrt(dd) + ramp[None, :]
    return softmax(np.where(causal_mask(n), s, -np.inf)) @ Vx
print("Breakage 3, one fixed shift of 0 and 16-bit exponentials, on scores that rise along the row to 16:")
def fixed(shift):
    nn = n; o = np.zeros((nn, dd)); l = np.zeros(nn)
    s = Qb @ Kb.T / np.sqrt(dd) + ramp[None, :]
    s = np.where(causal_mask(nn), s, -np.inf)
    with np.errstate(over="ignore", invalid="ignore"):
        p = np.exp((s - shift).astype(np.float16)).astype(float)
        o = p @ Vr; l = p.sum(1)
        return o / l[:, None]
ref3 = biased_ref(Qb, Kb, Vr)
for sh in (0, 8):
    with np.errstate(invalid="ignore"):
        r = fixed(sh); bad = int((~np.isfinite(r)).any(1).sum())
    print(f"  shift {sh}: rows with inf or nan {bad} of {n}" + ("" if bad else f" | largest error {np.abs(r - ref3).max():.1e}"))

print("\nPart 3. Checkpoint tests")
tests = []
big = rng.normal(size=(1024, 64)) * 3
Qt, Kt, Vt = big, rng.normal(size=(1024, 64)) * 3, rng.normal(size=(1024, 64))
plain = softmax(np.where(causal_mask(1024), Qt @ Kt.T / 8.0, -np.inf)) @ Vt
tests.append(("1. tiled equals plain to 1e-12, blocks of 64, random scores", np.abs(tiled_attention(Qt, Kt, Vt, 64) - plain).max() < 1e-12))
tests.append(("2. recovered share rows sum to one", np.allclose(P.sum(1), 1.0)))
rise = np.linspace(0, 12, 1024)[None, :]
plain_r = softmax(np.where(causal_mask(1024), Qt @ Kt.T / 8.0 + rise, -np.inf)) @ Vt
Kt_r = Kt  # the rise is added inside a broken and a correct copy below
def with_rise(mistake):
    nq = 1024; out = np.empty((nq, 64))
    for i0 in range(0, nq, 64):
        q = Qt[i0:i0 + 64] / 8.0; rows = np.arange(i0, i0 + 64)
        m = np.full(64, -np.inf); l = np.zeros(64); o = np.zeros((64, 64))
        for j0 in range(0, i0 + 64, 64):
            cols = np.arange(j0, j0 + 64)
            s = q @ Kt[j0:j0 + 64].T + rise[:, j0:j0 + 64]
            s = np.where(cols[None, :] <= rows[:, None], s, -np.inf)
            m_new = np.maximum(m, s.max(1)); first = np.isinf(m)
            sc = np.exp(m - m_new)
            if mistake == "no rescale": sc = np.where(first, 0.0, 1.0)
            p = np.exp(s - m_new[:, None]); l = l * sc + p.sum(1)
            o = o * (np.where(first, 0.0, 1.0)[:, None] if mistake == "sum only" else sc[:, None]) + p @ Vt[j0:j0 + 64]
            m = m_new
        out[i0:i0 + 64] = o / l[:, None]
    return out
tests.append(("3. Breakage 1 caught: no-rescale differs from plain on rising scores", np.abs(with_rise("no rescale") - plain_r).max() > 1e-3))
tests.append(("4. Breakage 2 caught: sum-only differs from plain on rising scores", np.abs(with_rise("sum only") - plain_r).max() > 1e-3))
tests.append(("5. correct code passes the same rising-score test", np.abs(with_rise(None) - plain_r).max() < 1e-12))
with np.errstate(over="ignore", invalid="ignore"):
    tests.append(("6. Breakage 3 caught: fixed shift 0 in 16-bit gives non-finite rows", not np.isfinite(fixed(0)).all()))
for name, ok in tests:
    print(f"  {name}: {'pass' if ok else 'FAIL'}")
print("all checkpoint tests pass:", all(ok for _, ok in tests))
