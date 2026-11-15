"""Chapter 8: two things one softmax head cannot do.  (1) It cannot say 'nothing here': shares
always sum to one.  (2) A weighted average forgets how much evidence there was.
Run from 04-code/ch08:  python3 limits.py"""
from _common import *
from tiny.attention import softmax, attention

print("(1) no abstention")
for label, row in [("one clear match", [4.0, 0, 0, 0]), ("weak matches", [0.2, 0.1, 0.0, 0.1]),
                   ("nothing matches", [-6.0, -6.1, -5.9, -6.0])]:
    print(f"  {label:>16}: scores {row} -> shares {np.round(softmax(np.array(row)), 3)} (sum {softmax(np.array(row)).sum():.0f})")
with_sink = np.array([3.0, -6.0, -6.1, -5.9, -6.0])
print("  add a 'sink' seat at the front with score 3:", np.round(softmax(with_sink), 4))

print("\n(2) a weighted average forgets how many")
Wq, Wk, Wv = W_q, W_k, W_v
sat, cat, the = X[2], X[1], X[0]
for label, rows in [("the cat sat", [the, cat, sat]), ("the cat cat cat sat", [the, cat, cat, cat, sat])]:
    out, A = attention(np.array(rows), Wq, Wk, Wv)
    print(f"  {label:>20}: last row's shares {np.round(A[-1], 3)} -> output {tuple(round(float(v), 3) for v in out[-1])}")
only_cat = [np.array([cat]), np.array([cat, cat, cat])]
for rows in only_cat:
    out, A = attention(rows, Wq, Wk, Wv, causal=False)
    print(f"  {len(rows)} cat row(s), no mask: every output row {tuple(round(float(v), 3) for v in out[0])}")
