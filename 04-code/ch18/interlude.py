"""Chapter 18, section 18.4 (Interlude): the chain rule for a table, and the slope of a softmax by picture.
Part 1: a number that feeds several places collects the sum of its nudges (K[cat@1] is read by four seats).
Part 2: the softmax's slopes for sat@2's three shares, by wiggling each score (h = 1e-6) and by the rule
        slope of share i in score j = P_i ([i = j] - P_j); the shortcut that never builds the table.
Part 3: the same for a row that is nearly one-hot, and for the@0's row of one share.  Nothing random."""
import _common
import numpy as np
from tiny.attention import softmax, softmax_backward
from _common import load_json, f4, row, SEATS
rc = load_json("recap"); j = load_json("backward_head")
S = np.array(rc["S"]); Q = np.array(rc["Q"]); dS = np.array(j["g"]["S"])
c = 1 / np.sqrt(2)
print("Part 1: one key, four readers.  cat@1's key is scored by cat@1, sat@2, on@3 and the@4, so")
parts = [dS[i, 1] * Q[i] * c for i in range(1, 5)]
for i, p in zip(range(1, 5), parts):
    print(f"  from {SEATS[i]:<6} dS[{SEATS[i]}, cat@1] x Q[{SEATS[i]}] / sqrt 2 = {f4(dS[i,1],5)} x {row(Q[i])} x 0.7071 = {row(p, 5)}")
print(f"  sum = {row(sum(parts), 5)}; the table rule dK = dS^T Q / sqrt 2 gives {row(np.array(j['g']['K'][1]), 5)}")

print("\nPart 2: sat@2's shares as its three scores move")
s = S[2, :3]; p = softmax(s)
print(f"  scores {row(s)} -> shares {row(p)}")
h = 1e-6
J = np.zeros((3, 3))
for jj in range(3):
    up, dn = s.copy(), s.copy(); up[jj] += h; dn[jj] -= h
    J[:, jj] = (softmax(up) - softmax(dn)) / (2 * h)
rule = np.diag(p) - np.outer(p, p)
print("  slope of share i (row) in score j (column), by wiggling:")
for i in range(3):
    print("   ", "  ".join(f4(v) for v in J[i]))
print(f"  rule P_i([i = j] - P_j): largest difference {np.abs(J - rule).max():.1e}")
print(f"  on the diagonal P(1 - P): {', '.join(f4(v) for v in np.diag(rule))}; off it -P_i P_j")
print(f"  each column adds to {f4(rule.sum(0).max(), 6)}: raising any score moves share around, never adds any")
dP = np.array(j["g"]["P"][2][:3])
print(f"  the nudge arriving at the shares, dP = {row(dP, 5)}")
print(f"  through the table: dS = J^T dP = {row(rule.T @ dP, 5)}")
print(f"  shortcut, no table: P x (dP - P . dP) = {row(softmax_backward(p, dP), 5)}")
print(f"  without the row-sum term, P x dP = {row(p * dP, 5)} (wrong: adds to {f4((p * dP).sum(), 5)})")

print("\nPart 3: where the slopes go to zero")
for name, sc in [("even, three scores of 0", np.zeros(3)), ("sat@2's row", s), ("one score 5 ahead", np.array([5.0, 0, 0])),
                 ("one score 10 ahead", np.array([10.0, 0, 0])), ("the@0's row, one share", np.array([S[0, 0]]))]:
    pp = softmax(sc); JJ = np.diag(pp) - np.outer(pp, pp)
    print(f"  {name:<26} shares {row(pp, 3):<28} largest slope {f4(np.abs(JJ).max())}")
print("  a share of exactly 1 has slope 0 in every score: the@0's scores get no nudge at all")
print("\nCheck yourself (answers in Appendix A)")
print(f"  1. three uses with slopes 0.2, -0.5, 0.1: nudge {0.2 - 0.5 + 0.1:+.1f}")
p2 = np.array([0.75, 0.25]); J2 = np.diag(p2) - np.outer(p2, p2)
print(f"  2. shares (0.75, 0.25): slope of the first share in its own score {J2[0, 0]:+.4f}, in the other score {J2[0, 1]:+.4f}")
p3 = np.array([0.5, 0.3, 0.2])
print(f"  3. shares (0.5, 0.3, 0.2), dP = (1, 1, 1): dS = {row(softmax_backward(p3, np.ones(3)))}")
