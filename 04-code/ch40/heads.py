"""Chapter 40, 40.8: what each head's attention looks like on ordinary text, and what removing it does.  Trained Tiny (Chapter 24),
100 windows of the pool (128 tokens) other than probe.py's.  Per head: share of attention on the first seat (the sink), on the previous seat,
on itself, mean entropy of the shares (nats; uniform over the visible seats would be the 'even' value), the rise in loss when the head's
output is zeroed before the output matrix (correct) and when the matching channel slice of the attention block's output is zeroed after
the output matrix (the naive version, S526).  Ends with the same two ablations on the 30 taught facts' log p(verb) in the fine-tuned model.
32-bit, seed 1865."""
from lib import *
import numpy as np
torch.set_num_threads(1); torch.manual_seed(SEED)
from _pool import corpora
from scipy.stats import spearmanr
tt = load(CH24); c = corpora(); pool = np.array([int(i) for i in c["pool"]])
rng = np.random.default_rng(SEED + 5); starts = rng.choice(len(pool) - 129, size=100, replace=False)
W = torch.tensor(np.stack([pool[s:s + 129] for s in starts])); X = W[:, :128]; Y = W[:, 1:]
NL, H = 12, 4
def loss(**kw):
    with torch.no_grad(): lg = forward(tt, X, **kw)["logits"]
    return float(torch.nn.functional.cross_entropy(lg.reshape(-1, 2048), Y.reshape(-1)))
base = loss(); print(f"mean loss on 100 windows: {base:.4f} nats")
with torch.no_grad(): A = forward(tt, X, want_att=True)["att"]
n = 128; rows = []
ent_even = float(np.mean([np.log(i + 1) for i in range(n)]))
sink = np.zeros((NL, H)); prev = np.zeros((NL, H)); selfs = np.zeros((NL, H)); ent = np.zeros((NL, H)); good = np.zeros((NL, H)); naive = np.zeros((NL, H))
for l in range(NL):
    a = A[l].mean(0)                                           # (h, n, n) averaged over windows
    for h in range(H):
        m = A[l][:, h]; sink[l, h] = float(m[:, 4:, 0].mean()); prev[l, h] = float(torch.diagonal(m, offset=-1, dim1=-2, dim2=-1)[:, 3:].mean())
        selfs[l, h] = float(torch.diagonal(m, dim1=-2, dim2=-1)[:, 4:].mean())
        ent[l, h] = float((-(m * torch.log(m + 1e-12)).sum(-1))[:, 4:].mean())
        good[l, h] = loss(ablate={(l, h)}) - base; naive[l, h] = loss(naive={(l, h)}) - base
    print(f"layer {l:2d} " + "  ".join(f"h{h}: sink {sink[l,h]:.2f} prev {prev[l,h]:.2f} self {selfs[l,h]:.2f} ent {ent[l,h]:.2f} dL {good[l,h]:+.3f}" for h in range(H)), flush=True)
print(f"entropy of attending evenly over every visible seat from seat 4 on: {np.mean([np.log(i + 1) for i in range(4, n)]):.2f} nats")
print(f"heads with sink share above 0.5: {(sink > 0.5).sum()} of 48; heads with previous-seat share above 0.3: {(prev > 0.3).sum()}")
print("heads by loss rise when zeroed (correct ablation), top 8: " + ", ".join(f"{i // H}.{i % H} {good.ravel()[i]:+.3f}" for i in np.argsort(-good.ravel())[:8]))
print("heads by loss rise (naive slice after the output matrix), top 8: " + ", ".join(f"{i // H}.{i % H} {naive.ravel()[i]:+.3f}" for i in np.argsort(-naive.ravel())[:8]))
t5a = set(np.argsort(-good.ravel())[:5]); t5b = set(np.argsort(-naive.ravel())[:5])
print(f"overlap of the two top-5 sets: {len(t5a & t5b)}; Spearman between the two effects over 48 heads: {spearmanr(good.ravel(), naive.ravel()).statistic:.3f}")
print(f"Spearman, sink share vs loss rise: {spearmanr(sink.ravel(), good.ravel()).statistic:.3f}; entropy vs loss rise: {spearmanr(ent.ravel(), good.ravel()).statistic:.3f}; previous-seat share vs loss rise: {spearmanr(prev.ravel(), good.ravel()).statistic:.3f}")
print(f"heads that raise the loss by under 0.01 nat when removed: {(good < 0.01).sum()} of 48; by over 0.1: {(good > 0.1).sum()}")
save_json("heads", dict(base=base, sink=sink.tolist(), prev=prev.tolist(), self=selfs.tolist(), ent=ent.tolist(), good=good.tolist(), naive=naive.tolist()))
