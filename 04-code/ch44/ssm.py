"""Chapter 44: a state-space layer in a few lines, checked three ways.  CPU, 64-bit numpy, seed 1865.
1. A linear time-invariant recurrence h_t = a*h_{t-1} + b*x_t, y_t = c.h_t gives the same output when computed as a
   recurrence (one step at a time, constant memory) and as a convolution with the kernel K_j = c.(a^j * b) (all at once).
2. A selective recurrence (a, b depend on x_t, as in Mamba) cannot be written as one fixed convolution: the best single kernel,
   fitted by least squares on 40 sequences, misses a large share of the output on 40 new ones; for the time-invariant layer it misses nothing.
3. Memory carried while generating: Tiny's key-value cache (12 layers, width 128, 32-bit) against a state-space stack of the
   same depth with 256 channels and a state of 16 per channel."""
import numpy as np
rng = np.random.default_rng(1865); N, T = 16, 64
a = np.exp(-rng.uniform(0.01, 0.5, N)); b = rng.normal(0, 1, N); c = rng.normal(0, 1, N); x = rng.normal(0, 1, T)
def recur(x, a_t, b_t):
    h = np.zeros(N); y = []
    for t in range(len(x)):
        h = a_t[t] * h + b_t[t] * x[t]; y.append(c @ h)
    return np.array(y)
y_rec = recur(x, [a] * T, [b] * T)
K = np.array([c @ (a ** j * b) for j in range(T)])
y_conv = np.array([sum(K[j] * x[t - j] for j in range(t + 1)) for t in range(T)])
print(f"1. time-invariant: largest |recurrence - convolution| over {T} steps = {np.abs(y_rec - y_conv).max():.2e}")
W = rng.normal(0, 0.5, N)
sel_a = lambda x: [np.exp(-np.log1p(np.exp(W * xt)) * 0.3) for xt in x]       # decay depends on the input (softplus step)
sel_b = lambda x: [b * (1 + 0.5 * np.tanh(xt)) for xt in x]
def toe(x): return np.array([[x[t - j] if t - j >= 0 else 0 for j in range(T)] for t in range(T)])
def best_kernel_test(sel):
    """Fit one kernel by least squares on 40 random sequences, then report the share of output variance it misses on 40 others."""
    def run(x): return recur(x, sel_a(x), sel_b(x)) if sel else recur(x, [a] * T, [b] * T)
    tr = [rng.normal(0, 1, T) for _ in range(40)]; te = [rng.normal(0, 1, T) for _ in range(40)]
    Kf = np.linalg.lstsq(np.vstack([toe(s) for s in tr]), np.concatenate([run(s) for s in tr]), rcond=None)[0]
    Y = np.concatenate([run(s) for s in te]); P = np.concatenate([toe(s) @ Kf for s in te])
    return ((Y - P) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
print(f"2. best single convolution kernel, share of output variance missed on 40 new sequences: "
      f"time-invariant {best_kernel_test(False):.2e}; selective {best_kernel_test(True):.3f}")
L, D = 12, 128; per_tok = 2 * L * D * 4
for n in (1024, 8192, 131072):
    print(f"3. Tiny's key-value cache at {n:7,} tokens: {per_tok * n:13,} bytes")
state = L * 256 * N * 4 + L * 256 * 3 * 4                                       # state plus a short convolution window of 4
print(f"3. a 12-layer state-space stack, 256 channels x state 16 (+ conv window 3): {state:,} bytes at any length "
      f"(Tiny's cache passes it at {state // per_tok + 1} tokens)")
