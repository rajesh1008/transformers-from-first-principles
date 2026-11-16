"""Chapter 8 Maths Interlude: softmax, turning scores into shares.
Run from 04-code/ch08:  python3 softmax_interlude.py"""
from _common import *
from tiny.attention import softmax

print("powers of e = 2.718...:")
for s in [-2, -1, 0, 1, 2, 3, 10, 100]:
    print(f"  e^{s:<4} = {np.exp(s):.6g}")

print("\nthree scores (2, 1, 0):")
s = np.array([2.0, 1.0, 0.0]); e = np.exp(s)
print("  exponentials", np.round(e, 3), "total", round(float(e.sum()), 3), "-> shares", np.round(e / e.sum(), 3))
print("  add 5 to every score:", np.round(softmax(s + 5), 3), "(unchanged)")
print("  double every score:  ", np.round(softmax(2 * s), 3), "(sharper)")
print("  halve every score:   ", np.round(softmax(s / 2), 3), "(flatter)")
print("  times 10:            ", np.round(softmax(10 * s), 5))

print("\nwhy not just divide each score by the total?")
s2 = np.array([2.0, -1.0, 0.0])
print("  scores (2, -1, 0): divided by their total 1 ->", s2 / s2.sum(), "(a negative 'share')")
s3 = np.array([1.0, -1.0, 0.0])
print("  scores (1, -1, 0): total is 0 ->", "division by zero")
print("  softmax of (2, -1, 0):", np.round(softmax(s2), 3))

print("\nsubtract the largest score first (the same shares, no overflow):")
big = np.array([1000.0, 999.0, 998.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(big) / np.exp(big).sum()
print("  naive exp(1000)/...:", naive)
print("  subtract 1000 first: ", np.round(softmax(big), 3))

print("\nshares of a row of six equal scores:", np.round(softmax(np.zeros(6)), 3))
print("one score 3 above five others:     ", np.round(softmax(np.array([3.0, 0, 0, 0, 0, 0])), 3))
print("a masked score (-inf) gets share:  ", softmax(np.array([1.0, 0.0, -np.inf]))[2])

print("\nCheck yourself")
print("  1. softmax(0, 0):", softmax(np.array([0.0, 0.0])))
print("  2. softmax(ln 3, 0) = (0.75, 0.25)?", np.round(softmax(np.array([np.log(3), 0.0])), 3))
print("  3. softmax(1, 2, 3) == softmax(101, 102, 103)?", bool(np.allclose(softmax(np.array([1., 2, 3])), softmax(np.array([101., 102, 103])))))
