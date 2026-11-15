"""Chapter 14, section 1: one layer, exploded.  Tiny's layer 0 (Chapter 13's pre-norm layer, seed 1865)
on the running sentence, in its six steps, with the shape and the length of ' sat''s row after each.
Run from 04-code/ch14:  python3 exploded.py"""
from _common import *
from tiny.model import Tiny

t = tok(); ids = sentence_ids(t); pieces = [piece(t, i) for i in ids]
m = Tiny()
X = m.embed(ids)
steps = m.blocks[0].explode(X)
k = pieces.index(" sat")
print(f"seed {SEED}; the running sentence as Tiny's tokens: {pieces} (ids {list(map(int, ids))})")
print(f"one layer, six steps, on the table of {len(ids)} rows; ' sat' is row {k}")
print("  step            | shape     | ' sat' length | first three numbers of ' sat'")
names = [("x", "row in"), ("ln_1", "1 norm"), ("attention", "2 attention"), ("add_1", "3 add"),
         ("ln_2", "4 norm"), ("feed_forward", "5 feed-forward"), ("add_2", "6 add (row out)")]
for key, label in names:
    T = steps[key]
    print(f"  {label:<15} | {str(T.shape):<9} | {length(T[k]):13.4f} | {r3(T[k][:3], 4)}")
print("\nbookkeeping: row out = row in + attention write + feed-forward write, exactly:",
      bool(np.allclose(steps["add_2"], steps["x"] + steps["attention"] + steps["feed_forward"])))
b = m.blocks[0]
print(f"learned numbers in the layer: ln_1 {b.ln_1.n_params} + attention {b.att.n_params:,} + ln_2 {b.ln_2.n_params} "
      f"+ feed-forward {b.ff.n_params:,} = {b.n_params:,}")
print("which steps let rows read each other? change only row 0 ('the' -> ' cat') and apply each step alone:")
X2 = X.copy(); X2[0] = X[1]
for label, f in [("norm (ln_1)", b.ln_1), ("attention", b.att), ("feed-forward", b.ff)]:
    moved = np.abs(f(X2)[1:] - f(X)[1:]).max()
    print(f"  {label:<13}: rows 1 to 6 move by at most {moved:.1e}")
