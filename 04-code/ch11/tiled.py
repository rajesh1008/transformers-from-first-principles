"""Chapter 11, sections 4 and 5: tiled attention in NumPy.  Queries in blocks of Br rows, keys and values
in blocks of Bc; each query block keeps a running max, a running sum and an unnormalised output, so no
n x n table is ever held.  Checks equality with the plain version, counts the largest table held, counts
blocks skipped by the causal mask and by a window, and times both on the CPU (fastest of five, recorded;
--measure to re-time).  Run from 04-code/ch11:  python3 tiled.py"""
from _common import *
import time
from tiny.attention import softmax, causal_mask

def plain(Q, K, V, causal=True):
    S = Q @ K.T / np.sqrt(Q.shape[1])
    if causal:
        S = np.where(causal_mask(len(Q)), S, -np.inf)
    return softmax(S) @ V

def tiled(Q, K, V, Br=64, Bc=64, causal=True, window=None, stats=None):
    n, d = Q.shape
    out = np.empty((n, V.shape[1]))
    largest = 0; done = skipped = 0
    for i0 in range(0, n, Br):
        q = Q[i0:i0 + Br] / np.sqrt(d); rows = np.arange(i0, min(i0 + Br, n))
        m = np.full(len(q), -np.inf); l = np.zeros(len(q)); o = np.zeros((len(q), V.shape[1]))
        for j0 in range(0, n, Bc):
            cols = np.arange(j0, min(j0 + Bc, n))
            if causal and cols[0] > rows[-1]:
                skipped += 1; continue                      # block wholly in the future
            if window is not None and cols[-1] <= rows[0] - window:
                skipped += 1; continue                      # block wholly outside the window
            s = q @ K[j0:j0 + Bc].T
            if causal:
                ok = cols[None, :] <= rows[:, None]
                if window is not None:
                    ok &= cols[None, :] > rows[:, None] - window
                s = np.where(ok, s, -np.inf)
            largest = max(largest, s.size); done += 1
            m_new = np.maximum(m, s.max(1))
            m_use = np.where(np.isinf(m_new), 0.0, m_new)   # a row with nothing allowed yet stays empty
            scale = np.exp(m - m_use)
            p = np.exp(s - m_use[:, None])
            l = l * scale + p.sum(1)
            o = o * scale[:, None] + p @ V[j0:j0 + Bc]
            m = m_new
        out[i0:i0 + Br] = o / l[:, None]
    if stats is not None:
        stats.update(largest=largest, done=done, skipped=skipped)
    return out

rng = np.random.default_rng(SEED)
n, d = 1024, 64
Q, K, V = (rng.normal(size=(n, d)) for _ in range(3))
st = {}
a, b = plain(Q, K, V), tiled(Q, K, V, stats=st)
print(f"n={n}, d={d}, blocks 64 x 64, seed {SEED}")
print(f"tiled equals plain: {bool(np.allclose(a, b))} | largest difference {np.abs(a - b).max():.1e}")
print(f"largest score table held: plain {n * n:,} numbers | tiled {st['largest']:,}")
print(f"blocks computed {st['done']} | skipped as wholly in the future {st['skipped']} of {(n // 64) ** 2}")
T = n // 64
print(f"  causal: computed fraction {st['done'] / T**2:.4f}; formula (T + 1) / 2T = {(T + 1) / (2 * T):.4f} with T = {T}")
st2 = {}
w = tiled(Q, K, V, window=256, stats=st2)
ref_w = softmax(np.where(causal_mask(n) & (np.arange(n)[None, :] > np.arange(n)[:, None] - 256),
                         Q @ K.T / np.sqrt(d), -np.inf)) @ V
print(f"window 256: equals masked plain {bool(np.allclose(w, ref_w))}; blocks computed {st2['done']} of {T**2}"
      f" ({st2['done'] / T**2:.1%}), skipped {st2['skipped']}")
for n2 in (4096, 16384):
    TT = n2 // 64
    full_b = TT * (TT + 1) // 2
    win_b = sum(min(i + 1, 256 // 64 + 1) for i in range(TT))
    print(f"  n={n2:,}: causal blocks {full_b:,} | window-256 blocks {win_b:,} | ratio {full_b / win_b:.1f}")

def measure():
    res = {}
    for nn in (1024, 2048, 4096):
        Qm, Km, Vm = (rng.normal(size=(nn, d)) for _ in range(3))
        for name, f in [("plain", lambda: plain(Qm, Km, Vm)), ("tiled 64", lambda: tiled(Qm, Km, Vm)),
                        ("tiled 256", lambda: tiled(Qm, Km, Vm, Br=256, Bc=256))]:
            best = 1e9
            for _ in range(5):
                t = time.perf_counter(); f(); best = min(best, time.perf_counter() - t)
            res[f"{name} {nn}"] = round(best * 1000, 1)
    return res
times = recorded("tiled_timing", measure)
print("\nCPU time, one head, causal, ms (fastest of 5, measured", times["measured"] + "):")
for nn in (1024, 2048, 4096):
    print(f"  n={nn:>5}: plain {times[f'plain {nn}']:>7} | tiled 64 x 64 {times[f'tiled 64 {nn}']:>7} | tiled 256 x 256 {times[f'tiled 256 {nn}']:>7}")
print("  plain holds an n x n table of 8-byte numbers:", ", ".join(f"{nn}: {nn * nn * 8 / 2**20:.0f} MiB" for nn in (1024, 2048, 4096)))
