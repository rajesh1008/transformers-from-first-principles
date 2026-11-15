"""Chapter 17, sections 17.1 and 17.3: one learned number, nudged by hand.

The number is the first of mat's two numbers in the hand Tiny's token table (Chapter 15), x = -1.
Because the table is tied, mat's row is also the row every seat's top row is scored against for the
word mat, and because mat sits at seat 5, which has no true next word and which no earlier seat can
read, changing x changes nothing in the forward pass except the score of mat at each seat:
    score of mat at seat s = top_s . (x, 1)
So the whole model, seen from x, is five fixed top rows and one number.  NumPy only; nothing random."""
import _common
import numpy as np
from hand import hand_tiny, IDS, WORDS
from _common import save_json

m = hand_tiny()
S, stream = m.forward(IDS, keep=True)
TOP = m.top_rows(stream[-1])                     # (6, 2): each seat's top row after the final norm
E = m.E.table
MAT = 4
SEATS = [0, 1, 2, 3, 4]                          # seats with a true next word
TRUE = IDS[1:]                                   # cat, sat, on, the, mat

def scores_at(x, s):
    e = E.copy(); e[MAT, 0] = x
    return TOP[s] @ e.T

def surprise_at(x, s, word):
    z = scores_at(x, s); z = z - z.max()
    return float(-(z[word] - np.log(np.exp(z).sum())))

def share_at(x, s, word):
    z = scores_at(x, s); e = np.exp(z - z.max()); return float(e[word] / e.sum())

def loss(x):
    return float(np.mean([surprise_at(x, s, TRUE[s]) for s in SEATS]))

def wiggle(f, x, h=1e-3):
    return (f(x + h) - f(x - h)) / (2 * h)

print("the@4's top row after the final norm:", _common.r3(TOP[4]))
print("mat's row (x, 1) with x =", E[MAT, 0])
one = lambda x: surprise_at(x, 4, MAT)
x = -1.0
print(f"\nPart 1: the surprise on 'mat' after the@4 alone")
print(f"  x = {x:+.3f}: score of mat {scores_at(x,4)[MAT]:+.3f}, share {share_at(x,4,MAT):.4f}, surprise {one(x):.4f}")
for h in (1.0, 0.1, 0.01, 0.001):
    print(f"  wiggle x by {h:<6}: surprise {one(x+h):.4f}; change / wiggle = {(one(x+h)-one(x))/h:+.4f}")
print(f"  central wiggle (h = 0.001): slope {wiggle(one, x):+.4f}")
p = share_at(x, 4, MAT); t4 = TOP[4, 0]
print(f"  rule from the Interlude: slope = (share - 1) x (the@4's first number) = ({p:.4f} - 1) x {t4:.4f} = {(p-1)*t4:+.4f}")

print("\n  three nudges by hand, learning rate 1 (x <- x - 1 x slope):")
rows = []
for step in range(4):
    sl = (share_at(x, 4, MAT) - 1) * t4
    rows.append(dict(step=step, x=x, share=share_at(x, 4, MAT), surprise=one(x), slope=sl))
    print(f"  step {step}: x {x:+.4f}  share {share_at(x,4,MAT):.4f}  surprise {one(x):.4f}  slope {sl:+.4f}")
    x = x - 1.0 * sl
print("  the surprise on one word has no bottom: at x = 20 the share is", round(share_at(20, 4, MAT), 6),
      "and the slope", round(wiggle(one, 20.0), 6))

print("\nPart 2: the loss, the average surprise over the five seats with a true next word")
x = -1.0
print(f"  x = -1: loss {loss(x):.4f} (Chapter 15's 3.3313)")
for s in SEATS:
    print(f"    seat {s} true next {WORDS[TRUE[s]]:<4} share of mat {share_at(x, s, MAT):.4f}  surprise {surprise_at(x, s, TRUE[s]):.4f}")
grid = np.linspace(-6, 6, 1201)
L = np.array([loss(g) for g in grid])
xbest = grid[L.argmin()]
# refine the bottom by gradient descent on the wiggled slope
xb = xbest
for _ in range(200):
    xb -= 0.5 * wiggle(loss, xb, 1e-5)
curv = (loss(xb + 1e-3) - 2 * loss(xb) + loss(xb - 1e-3)) / 1e-6
print(f"  bottom of the loss: x = {xb:.4f}, loss {loss(xb):.4f}; curvature there {curv:.4f}; 2 / curvature = {2/curv:.3f}")
sl = wiggle(loss, -1.0)
# the slope by the rule, summed over seats: for each seat, (share of mat - [mat is true]) x top_s[0], averaged
rule = np.mean([(share_at(-1.0, s, MAT) - (TRUE[s] == MAT)) * TOP[s, 0] for s in SEATS])
print(f"  slope of the loss at x = -1: wiggle {sl:+.5f}, rule {rule:+.5f}")

runs = {}
print("\nPart 3: gradient descent on the loss from x = -1, 12 steps, five learning rates")
for lr in (0.5, 2.0, 2/curv*0.98, 2/curv*1.05):
    x = -1.0; path = [x]
    for _ in range(12):
        x = x - lr * wiggle(loss, x, 1e-5); path.append(x)
    lab = f"{lr:.3f}"
    runs[lab] = dict(lr=lr, path=path, loss=[loss(p) for p in path])
    print(f"  lr {lr:6.3f}: x " + " ".join(f"{p:+.2f}" for p in path[:7]) + f" ... {path[-1]:+.3f}; loss {loss(path[0]):.4f} -> {loss(path[-1]):.4f}")
lr = 20.0; x = -1.0; path = [x]
for _ in range(12):
    x = x - lr * wiggle(loss, x, 1e-5); path.append(x)
print(f"  lr 20.000: x " + " ".join(f"{p:+.2f}" for p in path[:7]) + f" ... {path[-1]:+.3f}; loss {loss(path[0]):.4f} -> {loss(path[-1]):.4f}")
runs["20.000"] = dict(lr=20.0, path=path, loss=[loss(p) for p in path])
save_json("one_number", dict(top4=list(TOP[4]), rows=rows, grid=list(grid), L=list(L), xbest=xb, lbest=loss(xb),
                              curvature=curv, runs=runs))
