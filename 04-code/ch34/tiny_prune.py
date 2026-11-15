"""Chapter 34, 34.4: pruning Tiny without retraining.  Chapter 24's checkpoint (the one Chapters 25, 29 and 33 scored); every
2-D table except the token and position tables (the 2,359,296 numbers of the linear maps) has its smallest-|w| weights within each
output channel set to zero; bits per character on held-out Carroll in windows of 129 (32-bit: 1.9812, as Chapter 33 measured it).
Also whole feed-forward units removed (column of W_1, bias b_1, row of W_2) ranked by the product of norms.
CPU, one thread, seed 1865, 11 Nov 2026."""
from _common import *
import torch, math
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
from tiny.tokenizer import load_bpe
from _pool import corpora
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
t = load_bpe(TINY_BPE); ids = [int(i) for i in corpora()["held_b"]]
orig = {k: v.clone() for k, v in tt.t.items()}
mats = [k for k, v in orig.items() if v.dim() == 2 and k not in ("E", "P")]
print("tables:", sorted({k.split('.')[-1] for k in mats}), "| linear-map numbers", sum(orig[k].numel() for k in mats))
def bpc():
    tot, chars = 0.0, 0
    for a in range(0, len(ids) - 1, 129):
        w = ids[a:a + 129]
        if len(w) < 2: break
        with torch.no_grad(): lp = torch.log_softmax(tt.scores(w[:-1]).double(), -1)
        tot += float(-lp[torch.arange(len(w) - 1), torch.tensor(w[1:])].sum()); chars += len(t.decode(w[1:]))
    return tot / math.log(2) / chars
def reset():
    for k in orig: tt.t[k] = orig[k].clone()
base = bpc(); print(f"32-bit: {base:.4f}"); R = {"base": base}
print("\nsmallest weights within each output channel set to zero (linear maps only)")
for frac in (0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
    reset()
    for k in mats:
        w = orig[k]; kk = int(w.shape[0] * frac)
        idx = w.abs().argsort(0)[:kk]; m = torch.ones_like(w); m.scatter_(0, idx, 0.0); tt.t[k] = w * m
    v = bpc(); R[f"w {frac}"] = v; print(f"  {int(frac * 100):2d}% zeroed: {v:.4f} ({v - base:+.4f})", flush=True)
print("\nwhole feed-forward units removed in every layer")
for frac in (0.1, 0.25, 0.5):
    reset()
    for l in range(tt.config["n_layers"]):
        p = f"h.{l}.ff."; W1, b1, W2 = orig[p + "W_1"], orig[p + "b_1"], orig[p + "W_2"]
        score = W1.norm(dim=0) * W2.norm(dim=1); drop = score.argsort()[:int(W1.shape[1] * frac)]
        a, b, c = W1.clone(), b1.clone(), W2.clone(); a[:, drop] = 0; b[drop] = 0; c[drop, :] = 0
        tt.t[p + "W_1"], tt.t[p + "b_1"], tt.t[p + "W_2"] = a, b, c
    v = bpc(); R[f"u {frac}"] = v; print(f"  remove {int(frac * 100):2d}% of the units: {v:.4f} ({v - base:+.4f})", flush=True)
save_json("tiny_prune", R)
