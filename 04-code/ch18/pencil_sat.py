"""Chapter 18, section 18.5: sat@2's row of the backward pass in plain Python, as on paper, every number rounded
to five places before it is used (the pencil), against the 64-bit chain of backward_head.py.  The checkpoint is
the nudge on sat's query.  Nothing random."""
import _common
from _common import load_json
j = load_json("backward_head"); rc = load_json("recap")
R = lambda v: round(v, 5)
P = [R(v) for v in rc["P"][2][:3]]                     # sat@2 sees the@0, cat@1, sat@2
dO = [R(v) for v in j["dY"][2]]
V = [[R(v) for v in rc["V"][s]] for s in range(3)]
K = [[R(v) for v in rc["K"][s]] for s in range(3)]
print("sat@2 by pencil (five places at every step)")
print(f"  shares P = {P}; arriving nudge dO = {dO}")
dP = [R(dO[0] * V[j][0] + dO[1] * V[j][1]) for j in range(3)]
for jj, name in enumerate(["the@0", "cat@1", "sat@2"]):
    print(f"  dP[{name}] = dO . V[{name}] = {dO[0]} x {V[jj][0]} + {dO[1]} x {V[jj][1]} = {dP[jj]}")
rs = R(sum(P[i] * dP[i] for i in range(3)))
print(f"  row-sum term = sum of P x dP = {' + '.join(f'{P[i]} x {dP[i]}' for i in range(3))} = {rs}")
dS = [R(P[i] * (dP[i] - rs)) for i in range(3)]
for jj, name in enumerate(["the@0", "cat@1", "sat@2"]):
    print(f"  dS[{name}] = {P[jj]} x ({dP[jj]} - ({rs})) = {dS[jj]}")
print(f"  check: the three add to {R(sum(dS))}")
c = R(1 / 2 ** 0.5)
dQ = [R(c * sum(dS[i] * K[i][d] for i in range(3))) for d in range(2)]
print(f"  dQ[sat] = (1/sqrt 2) x (dS[the@0] K[the@0] + dS[cat@1] K[cat@1] + dS[sat@2] K[sat@2]) = {dQ}")
code = j["g"]["Q"][2]
print(f"  64-bit chain: ({code[0]:+.6f}, {code[1]:+.6f}); pencil off by {max(abs(dQ[0]-code[0]), abs(dQ[1]-code[1])):.1e}")
print("\nwhat sat@2's row sends elsewhere: its share of every key's nudge, dK[j] gets dS[sat, j] x Q[sat] / sqrt 2")
Qs = [R(v) for v in rc["Q"][2]]
for jj, name in enumerate(["the@0", "cat@1", "sat@2"]):
    print(f"  to {name}'s key: {dS[jj]} x {Qs} x {c} = {[R(dS[jj] * q * c) for q in Qs]}")
