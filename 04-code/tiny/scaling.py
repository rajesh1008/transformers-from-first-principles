"""Chapter 23: fitting scaling laws to small runs.  NumPy only.

    compute(N, D)              training operations, about 6 N D (Chapter 17: 2 forward + 4 backward per number per token)
    fit_line(x, y)             least-squares straight line on log-log axes: y = A x^(-alpha), returns (A, alpha)
    fit_floor(x, y)            y = E + A x^(-alpha): E found by a fine search below min(y), A and alpha by the line fit
                               of log(y - E) on log x, keeping the E whose curve misses the measured y least (squared)
    predict(fit, x)            the fitted curve at x
    leave_one_out(x, y)        refit with each point removed; returns the fits and each removed point's miss
    isoflop_minimum(N, L)      parabola through (log N, L) at one compute budget; the N at its bottom
    doublings(alpha)           the factor by which the reducible loss shrinks per doubling of x: 2^(-alpha)

Every function takes plain lists or arrays and returns floats or dicts, so a fit can be checked by hand."""
import numpy as np

def compute(N, D):
    return 6.0 * np.asarray(N, float) * np.asarray(D, float)

def fit_line(x, y):
    """Least squares on (log x, log y): the straight line with the smallest sum of squared misses in log y."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    slope, icept = np.polyfit(lx, ly, 1)
    return dict(E=0.0, A=float(np.exp(icept)), alpha=float(-slope))

def _line_given_E(lx, y, E):
    slope, icept = np.polyfit(lx, np.log(y - E), 1)
    A, alpha = np.exp(icept), -slope
    miss = y - (E + A * np.exp(-alpha * lx))
    return float(A), float(alpha), float((miss ** 2).sum())

def fit_floor(x, y, E_max=None, steps=4000):
    """y = E + A x^(-alpha).  E is searched on a grid from 0 to just below min(y) (or E_max); for each E the
    line fit gives A and alpha; the E with the smallest sum of squared misses in y wins.  Returns a dict with
    E, A, alpha, sse (sum of squared misses) and at_edge (True if the best E is the largest allowed, which means
    the data cannot tell the floor from the curve)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    lx = np.log(x)
    top = min(y.min() - 1e-4, E_max if E_max is not None else np.inf)
    grid = np.linspace(0.0, top, steps)
    best = None
    for E in grid:
        A, alpha, sse = _line_given_E(lx, y, E)
        if best is None or sse < best[3]:
            best = (E, A, alpha, sse)
    E, A, alpha, sse = best
    return dict(E=float(E), A=A, alpha=alpha, sse=sse, at_edge=bool(E >= grid[-2]), at_zero=bool(E <= grid[1]))

def predict(fit, x):
    return fit["E"] + fit["A"] * np.asarray(x, float) ** (-fit["alpha"])

def leave_one_out(x, y, floor=True):
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        f = fit_floor(x[keep], y[keep]) if floor else fit_line(x[keep], y[keep])
        out.append(dict(left_out=i, fit=f, miss=float(y[i] - predict(f, x[i]))))
    return out

def isoflop_minimum(N, L):
    """Fit L = a (log N)^2 + b log N + c and return the N at the bottom (exp(-b / 2a)) with the parabola; if the
    parabola opens downwards (a <= 0) there is no bottom inside the data and None is returned for N_opt."""
    lN = np.log(np.asarray(N, float))
    a, b, c = np.polyfit(lN, np.asarray(L, float), 2)
    if a <= 0:
        return dict(N_opt=None, a=float(a), b=float(b), c=float(c))
    l0 = -b / (2 * a)
    return dict(N_opt=float(np.exp(l0)), L_opt=float(c - b * b / (4 * a)), a=float(a), b=float(b), c=float(c),
                inside=bool(lN.min() <= l0 <= lN.max()))

def doublings(alpha):
    return 2.0 ** (-alpha)
