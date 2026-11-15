"""Chapter 9: what GPT-2's 144 heads do on the running sentence, sorted by a simple rule.
Uses the shares saved by Chapter 8's gpt2_head.py (04-code/ch08/outputs/gpt2_shares.npy).
Rows 1 to 5 only (row 0 can only look at itself).  A head is called
  previous-token  if its rows give on average more than half their share to the seat just before,
  first-token     if they give more than half to the first token (the attention sink of Chapter 8),
  self            if they give more than half to themselves,
  mixed           otherwise.
Six tokens is a tiny sample: a label here is a description of this sentence, not of the head.
Run from 04-code/ch09:  python3 heads_gallery.py"""
from _common import *
S = np.load(os.path.join(HERE, "..", "ch08", "outputs", "gpt2_shares.npy"))     # (12, 12, 6, 6)
r = np.arange(1, 6)
prev = S[:, :, r, r - 1].mean(-1); first = S[:, :, r, 0].mean(-1); self_ = S[:, :, r, r].mean(-1)
label = np.full((12, 12), "mixed", dtype=object)
label[self_ > 0.5] = "self"; label[first > 0.5] = "first-token"; label[prev > 0.5] = "previous-token"
for name in ("first-token", "previous-token", "self", "mixed"):
    print(f"{name:>15}: {(label == name).sum():>3} of 144")
print("\nper layer (first-token / previous-token / self / mixed):")
for l in range(12):
    c = [(label[l] == n).sum() for n in ("first-token", "previous-token", "self", "mixed")]
    print(f"  layer {l:>2}: " + " / ".join(str(int(v)) for v in c))
print("\nprevious-token heads:", ", ".join(f"{l}.{h} ({prev[l, h]:.2f})" for l, h in zip(*np.where(label == "previous-token"))))
print("self heads:", ", ".join(f"{l}.{h} ({self_[l, h]:.2f})" for l, h in zip(*np.where(label == "self"))))
print("\nlayer 5, the twelve heads, rows 1-5: share to first token / to previous / to self")
for h in range(12):
    print(f"  5.{h:<2} {first[5, h]:.2f} / {prev[5, h]:.2f} / {self_[5, h]:.2f}   {label[5, h]}")
