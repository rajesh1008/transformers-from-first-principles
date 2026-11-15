"""Chapter 35, 35.1: Tiny one token past the edge.  Chapter 24's Tiny was trained on windows of 129 tokens (positions 0 to 127
of its 1,024-row position table ever received a nudge).  Score Through the Looking-Glass (held out) in windows of 512 tokens
and report the mean loss by position; compare with scoring every token with at most the last 128 tokens in view (a sliding
window, one forward pass per token).  CPU, one thread, 32-bit."""
from _common import *
import torch, math, numpy as np
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
from _pool import corpora
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
g = [int(i) for i in corpora()["glass"]]
N, K = 512, 12                      # 12 windows of 512 from the start of the novel
L = np.zeros((K, N - 1))
for w in range(K):
    ids = g[w * N:(w + 1) * N]
    with torch.no_grad(): lp = torch.log_softmax(tt.scores(ids[:-1]).double(), -1)
    L[w] = -lp[torch.arange(N - 1), torch.tensor(ids[1:])].numpy()
B = [(0, 15), (16, 63), (64, 127), (128, 191), (192, 255), (256, 383), (384, 510)]
R = {"windows": K, "len": N}
print(f"Tiny on {K} windows of {N} Looking-Glass tokens: mean loss (nats) of the token predicted at each position")
for a, b in B:
    v = float(L[:, a:b + 1].mean()); R[f"full {a}-{b}"] = v; print(f"  positions {a:3d} to {b:3d}: {v:.4f}")
# position-table rows: how far did training move them?  Untrained rows keep their seed-1865 starting values.
from tiny.model import Tiny
P0 = torch.tensor(Tiny().P.table, dtype=torch.float32)
P = tt.t["P"]; moved = (P - P0).norm(dim=1)
for a, b in ((0, 127), (128, 255), (256, 1023)):
    v = float(moved[a:b + 1].mean()); R[f"P moved {a}-{b}"] = v; print(f"  position rows {a} to {b}: mean distance moved from their starting values {v:.4f}")
# sliding window of 128: token t predicted from tokens max(0, t-127) .. t
S = np.zeros((K, N - 1))
for w in range(K):
    ids = g[w * N:(w + 1) * N]
    for t in range(N - 1):
        ctx = ids[max(0, t - 127):t + 1]
        with torch.no_grad(): lp = torch.log_softmax(tt.scores(ctx).double()[-1], -1)
        S[w, t] = -float(lp[ids[t + 1]])
print("sliding window of 128 (every prediction sees at most the last 128 tokens, positions renumbered from 0):")
for a, b in B:
    v = float(S[:, a:b + 1].mean()); R[f"slide {a}-{b}"] = v; print(f"  positions {a:3d} to {b:3d}: {v:.4f}")
R["full mean"], R["slide mean"] = float(L.mean()), float(S.mean())
print(f"mean over all positions: full window {L.mean():.4f}, sliding window {S.mean():.4f}")
save_json("tiny_past_edge", R)
