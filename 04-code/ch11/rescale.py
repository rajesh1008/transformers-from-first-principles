"""Chapter 11, section 7: rescaling, and two ways to do less of it.  In online softmax the running output
is rescaled whenever a row's running max grows.  (1) Conditional rescaling: only rescale when the max has
grown by more than a threshold tau (here log(256), as a 2026 kernel does in base 2); otherwise keep the old
max as the shift.  Exact after the final division, as long as nothing overflows.  (2) A fixed shift chosen in
advance: no rescaling at all, but the numbers can overflow or lose precision in 16-bit arithmetic.
Run from 04-code/ch11:  python3 rescale.py"""
from _common import *
from tiny.attention import softmax

rng = np.random.default_rng(SEED)
n, d, Bc = 4096, 64, 64
Q = rng.normal(size=(256, d)); K = rng.normal(size=(n, d)); V = rng.normal(size=(n, d))
S = Q @ K.T / np.sqrt(d)
ref = softmax(S) @ V

def online(S, V, tau=0.0):
    m = np.full(len(S), -np.inf); l = np.zeros(len(S)); o = np.zeros((len(S), V.shape[1])); resc = 0; blocks = 0
    for j0 in range(0, S.shape[1], Bc):
        s = S[:, j0:j0 + Bc]; mb = s.max(1)
        grow = mb > m + tau                                  # rescale only these rows
        m_new = np.where(grow, np.maximum(m, mb), m)
        if j0 > 0:
            resc += int(grow.sum()); blocks += len(S)
        scale = np.where(np.isinf(m), 0.0, np.exp(np.where(np.isinf(m), 0, m) - m_new))
        p = np.exp(s - m_new[:, None])
        l = l * scale + p.sum(1); o = o * scale[:, None] + p @ V[j0:j0 + Bc]; m = m_new
    return o / l[:, None], resc, blocks, float(np.exp((S - m[:, None]).max()))

print(f"256 query rows x {n:,} keys, width {d}, key blocks of {Bc}, random scores (seed {SEED})")
for tau, name in [(0.0, "always rescale when the max grows"), (np.log(256), "rescale only if it grows by > log 256")]:
    out, resc, blocks, peak = online(S, V, tau)
    print(f"  {name:<40}: rescales {resc:>5,} of {blocks:,} row-blocks ({resc / blocks:.2%}) |"
          f" largest exp held {peak:8.2f} | error {np.abs(out - ref).max():.1e}")

print("\nthe same with scores that grow along the row (a strongly recency-biased head):")
S2 = S + np.linspace(0, 12, n)[None, :]
ref2 = softmax(S2) @ V
for tau, name in [(0.0, "always rescale"), (np.log(256), "threshold log 256")]:
    out, resc, blocks, peak = online(S2, V, tau)
    print(f"  {name:<40}: rescales {resc:>5,} of {blocks:,} ({resc / blocks:.2%}) | largest exp held {peak:10.2f} | error {np.abs(out - ref2).max():.1e}")

print("\nfixed shift (no running max at all), 16-bit exponentials:")
for shift_name, c in [("shift = true row max", S.max(1, keepdims=True)), ("shift = 8 for every row", 8.0), ("shift = 0", 0.0)]:
    with np.errstate(over="ignore", invalid="ignore"):
        E = np.exp((S - c).astype(np.float16)).astype(np.float64)
        P = E / E.sum(1, keepdims=True)
    bad = int((~np.isfinite(P)).any(1).sum())
    err = np.nanmax(np.abs(P - softmax(S)))
    print(f"  {shift_name:<26}: rows with overflow {bad:>3} | largest share error {err:.1e}")
for c in (8.0, 0.0):
    with np.errstate(over="ignore", invalid="ignore"):
        E = np.exp((S2 - c).astype(np.float16)).astype(np.float64)
    print(f"  recency-biased scores, shift {c:.0f}: rows with overflow {int((~np.isfinite(E)).any(1).sum())} of 256"
          f" (largest score {S2.max():.1f}; float16 exp overflows above 11.09)")
