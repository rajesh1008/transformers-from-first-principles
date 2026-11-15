"""Chapter 17, section 17.2 and Signature 6, the Loss Hill: the hand Tiny's loss (average surprise over
the running sentence's five predictions) as a surface over mat's two numbers, and a ball rolled down it
by gradient descent.  Mat's row changes only mat's score at each seat (see one_number.py), so the
surface can be computed exactly on a grid.  The nudge is computed by the rule of section 17.3, checked
against wiggling at the start.  NumPy only; nothing random."""
import _common
import numpy as np
from hand import hand_tiny, IDS, WORDS
from _common import save_json

m = hand_tiny()
S, stream = m.forward(IDS, keep=True)
TOP = m.top_rows(stream[-1])[:5]
E = m.E.table.copy()
MAT = 4
TRUE = np.array(IDS[1:])

def loss_and_nudge(row):
    e = E.copy(); e[MAT] = row
    Z = TOP @ e.T
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z); P /= P.sum(1, keepdims=True)
    L = float(-np.log(P[np.arange(5), TRUE]).mean())
    # slope of each seat's surprise in mat's score is (share of mat - 1 if mat is the true word else 0);
    # mat's score is top_s . row, so the nudge on row is that slope times top_s (nudges multiply), averaged
    coef = P[:, MAT] - (TRUE == MAT)
    return L, (coef[:, None] * TOP).mean(0)

start = np.array([-1.0, 1.0])
L0, g0 = loss_and_nudge(start)
h = 1e-6
wig = [(loss_and_nudge(start + h * np.eye(2)[i])[0] - loss_and_nudge(start - h * np.eye(2)[i])[0]) / (2 * h) for i in range(2)]
print(f"start: mat's row (-1, 1), loss {L0:.4f}; nudge by the rule {_common.r3(g0, 5)}, by wiggling {_common.r3(wig, 5)}")
xs = np.linspace(-6, 4, 201); ys = np.linspace(-2, 10, 241)
G = np.array([[loss_and_nudge(np.array([a, b]))[0] for a in xs] for b in ys])
i, j = np.unravel_index(G.argmin(), G.shape)
print(f"grid bottom near ({xs[j]:.2f}, {ys[i]:.2f}), loss {G.min():.4f}")
paths = {}
for lr in (1.0, 6.5, 7.5, 14.0):
    p = start.copy(); path = [p.copy()]; Ls = [L0]
    for _ in range(400):
        L, g = loss_and_nudge(p); p = p - lr * g; path.append(p.copy()); Ls.append(loss_and_nudge(p)[0])
    paths[str(lr)] = dict(path=[list(q) for q in path[:121]], loss=Ls)
    first = next((k for k, v in enumerate(Ls) if v < 2.8668), None)
    print(f"lr {lr:>4}: 1 step {_common.r3(path[1])} loss {Ls[1]:.4f}; 10 steps loss {Ls[10]:.4f}; 100 steps {_common.r3(path[100])} loss {Ls[100]:.4f}; "
          f"400 steps loss {Ls[400]:.4f}; first step within 0.01 of the bottom: {first}")
# curvature (the hill's steepest bend) at the bottom, by wiggling the nudge
p = start.copy()
for _ in range(20000):
    p = p - 6.5 * loss_and_nudge(p)[1]
H = np.array([(loss_and_nudge(p + 1e-4 * np.eye(2)[k])[1] - loss_and_nudge(p - 1e-4 * np.eye(2)[k])[1]) / 2e-4 for k in range(2)])
ev = np.linalg.eigvalsh((H + H.T) / 2)
print(f"bottom: mat's row {_common.r3(p, 4)}, loss {loss_and_nudge(p)[0]:.4f}; bends {ev.round(4)}; 2 / steepest bend = {2/ev.max():.2f}")
e = E.copy(); e[MAT] = p
Z = TOP @ e.T; P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
print("share of mat at each seat at the bottom:", _common.r3(P[:, MAT]), "; share of the true word:", _common.r3(P[np.arange(5), TRUE]))
save_json("loss_hill", dict(xs=list(xs), ys=list(ys), G=G.tolist(), paths=paths, bottom=list(p), bends=list(ev)))
