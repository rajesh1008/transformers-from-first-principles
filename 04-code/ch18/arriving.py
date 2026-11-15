"""Chapter 18, sections 18.2 and 18.3: the loss at the end, and the nudge that arrives at layer 0's head.
Part 1: at the output, the nudge on each seat's scores is (share - 1 for the true word, share for every rival) / 5.
Part 2: back through the tied output table and the final RMSNorm, by hand for seat 4, checked against autograd.
Part 3: the nudge arriving at the head's output in layer 0, from autograd, checked by wiggling each of its ten
numbers (a central wiggle of 1e-6 added to the head's output).  64-bit; nothing random."""
import _common
import numpy as np, torch
from hand import hand_tiny, IDS, WORDS
from _common import traced, npy, SEATS, f4, row, save_json

m = hand_tiny()
L, k = traced(m, IDS); L.backward()
print(f"loss {L.item():.4f}: the average surprise of five seats")
S = npy(k["scores"]); dS = npy(k["scores"].grad)
P = np.exp(S - S.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
print("\nPart 1: the nudge on each seat's twelve scores, by the rule (share - [true word]) / 5")
target = IDS[1:]
for s in range(5):
    t = target[s]
    rule = P[s].copy(); rule[t] -= 1; rule /= 5
    print(f"  {SEATS[s]:<6} true next {WORDS[t]:<4} share {P[s, t]:.4f}  nudge on its score {f4(rule[t])}"
          f"  largest rival {WORDS[int(np.argsort(-P[s])[0 if np.argmax(P[s]) != t else 1])]:<6}"
          f"  rule = autograd to {np.abs(rule - dS[s]).max():.1e}")
print(f"  each seat's twelve nudges add to {f4(dS.sum(-1).max(), 6)} (the shares add to one)")

E = m.tables()["E"]; z = npy(k["z"]); dz = npy(k["z"].grad)
print("\nPart 2: through the tied output, scores = z E^T, so dz = dscores E")
print(f"  seat 4: dz = {row(dS[4] @ E)}; autograd {row(dz[4])}")
top = npy(k["top"]); dtop = npy(k["top"].grad); g = m.tables()["ln_f.gain"]
x = top[4]; r = np.sqrt((x ** 2).mean() + 1e-5); u = dz[4] * g
hand = u / r - x * (u @ x) / (len(x) * r ** 3)
print(f"  through ln_f (RMSNorm) at seat 4: x = {row(x)}, rms {r:.4f}")
print(f"    dx = u / rms - x (u . x) / (2 rms^3), u = dz x gain: {row(hand)}; autograd {row(dtop[4])}")
print(f"    the part of u along x is removed: (dx . x) = {f4(hand @ x, 6)}")

dY = npy(k["Y"].grad).squeeze()
print("\nPart 3: the nudge arriving at layer 0's head output, dY, one row per seat")
h = 1e-6
wig = np.zeros_like(dY)
for s in range(5):
    for j in range(2):
        off = torch.zeros(5, 2, dtype=torch.float64); off[s, j] = h
        up = traced(m, IDS, offset=off)[0].item()
        off[s, j] = -h
        dn = traced(m, IDS, offset=off)[0].item()
        wig[s, j] = (up - dn) / (2 * h)
for s in range(5):
    print(f"  {SEATS[s]:<6} autograd {row(dY[s], 5)}   wiggled {row(wig[s], 5)}")
print(f"  largest difference {np.abs(dY - wig).max():.1e}")
dres = npy(k["x"].grad)
print("  the head's output is added to the residual stream, so dY is the nudge on the stream after the add;")
print(f"  the row entering layer 0 gets dY back unchanged on the skip path plus what comes through the head")
save_json("arriving", dict(loss=L.item(), dscores=dS.tolist(), dY=dY.tolist(), wig=wig.tolist(), dz4=dz[4].tolist(),
                           dtop4=dtop[4].tolist(), top4=x.tolist()))
