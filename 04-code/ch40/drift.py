"""Chapter 40, 40.9: what did fine-tuning move?  Base Tiny (Chapter 24) against the fine-tuned copy (Chapter 38, 600 steps at 3e-4) on the same
inputs: relative change of each layer's rows, ||r_ft - r_base|| / ||r_base||, averaged over the seats of 100 pool windows and over the site
seats of the 30 fact prompts; and the share of table change: per table, ||W_ft - W_base|| / ||W_base|| for the embedding table, the four
attention maps and two feed-forward maps averaged over layers.  Also: the word-start, next-word-start and previous-token probes of probe.py's
design trained on the fine-tuned model's rows (same windows), to see whether general knowledge in the rows moved.  32-bit, seed 1865."""
from lib import *
import numpy as np
torch.set_num_threads(1)
exec(open("facts_common.py").read())
from _pool import corpora
c = corpora(); pool = np.array([int(i) for i in c["pool"]]); rng = np.random.default_rng(SEED + 5); st = rng.choice(len(pool) - 129, size=100, replace=False)
X = torch.tensor(np.stack([pool[s:s + 128] for s in st]))
with torch.no_grad(): rb = forward(base, X)["resid"]; rf = forward(ft, X)["resid"]
rel = [float(((f - b).norm(dim=-1) / b.norm(dim=-1)).mean()) for b, f in zip(rb, rf)]
print("relative change of the rows by layer (pool windows): " + " ".join(f"{r:.3f}" for r in rel))
with torch.no_grad():
    sb = [forward(base, P[a]["ids"][:P[a]["site"] + 1])["resid"] for a in names]; sf = [forward(ft, P[a]["ids"][:P[a]["site"] + 1])["resid"] for a in names]
relf = [float(np.mean([float((f[l][0, -1] - b[l][0, -1]).norm() / b[l][0, -1].norm()) for b, f in zip(sb, sf)])) for l in range(13)]
print("relative change of the rows by layer (the 30 sites):   " + " ".join(f"{r:.3f}" for r in relf))
def tab(name): return float((ft.t[name] - base.t[name]).norm() / base.t[name].norm())
groups = {"embedding E": ["E"], "seat table P": ["P"], "attention maps": [f"h.{l}.att.W_{m}" for l in range(12) for m in "qkvo"],
          "feed-forward maps": [f"h.{l}.ff.W_{m}" for l in range(12) for m in "12"], "norm gains": [f"h.{l}.ln_{m}.gain" for l in range(12) for m in "12"]}
for g, ns in groups.items(): print(f"relative change of {g}: mean {np.mean([tab(n) for n in ns]):.4f}")
print("by layer, relative change of the attention maps: " + " ".join(f"{np.mean([tab(f'h.{l}.att.W_{m}') for m in 'qkvo']):.4f}" for l in range(12)))
print("by layer, relative change of the feed-forward maps: " + " ".join(f"{np.mean([tab(f'h.{l}.ff.W_{m}') for m in '12']):.4f}" for l in range(12)))
R = dict(rel_pool=rel, rel_sites=relf, groups={g: float(np.mean([tab(n) for n in ns])) for g, ns in groups.items()},
         att_by_layer=[float(np.mean([tab(f'h.{l}.att.W_{m}') for m in 'qkvo'])) for l in range(12)], ff_by_layer=[float(np.mean([tab(f'h.{l}.ff.W_{m}') for m in '12'])) for l in range(12)])
save_json("drift", R)
