"""Chapter 27: the share and rank of the running sentence's last word after "the cat sat on the" (2,500-step Tiny)."""
from _common import *
import torch; torch.set_num_threads(1)
from tiny.eval import load
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")); t = tok()
ids = t.encode("the cat sat on the"); full = t.encode("the cat sat on the mat")
print("full sentence tokens", full, [t.decode([i]) for i in full])
with torch.no_grad(): p = torch.softmax(tt.scores(ids)[-1].double(), -1).numpy()
m = full[len(ids)]; rank = int((p > p[m]).sum()) + 1
print(f"share of {t.decode([m])!r} after 'the cat sat on the': {p[m]:.4f}, rank {rank} of {len(p)}")
