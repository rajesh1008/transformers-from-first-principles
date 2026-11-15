"""Chapter 17, sections 17.5 and 17.7: every nudge in the hand Tiny on the running sentence.

PyTorch autograd gives the nudge for all 112 learned numbers in one backward pass; wiggling each number
in NumPy (224 forward passes) gives the same nudges independently.  Then: the nudge each of the six
seats' rows receives (the row entering layer 0, token row plus position row), how the token table's
row for 'the' collects its nudge from three places (seat 0, seat 4 and the tied output), and the nudges
on layer 0's attention tables.  64-bit throughout; nothing random."""
import _common
import time
import numpy as np
import torch
from hand import hand_tiny, IDS, WORDS
from tiny.torch_tiny import TorchTiny
from tiny.train import wiggle_grads
from _common import save_json, r3

m = hand_tiny()
tt = TorchTiny.from_numpy(m)
t0 = time.perf_counter(); L, G = tt.grads(IDS); t_auto = time.perf_counter() - t0
t0 = time.perf_counter(); W = wiggle_grads(m, IDS, h=1e-6); t_wig = time.perf_counter() - t0
diffs = {k: float(np.abs(G[k] - W[k]).max()) for k in G}
n = sum(v.size for v in G.values())
print(f"hand Tiny: {n} learned numbers, loss {L:.4f}")
print(f"autograd: one forward and one backward pass, {t_auto*1000:.1f} ms; wiggling: {2*n} forward passes, {t_wig*1000:.0f} ms")
worst = max(diffs, key=diffs.get)
print(f"largest difference between the two, over all {n} nudges: {diffs[worst]:.2e} (table {worst})")
big = sorted(((abs(float(v)), k, i) for k, a in G.items() for i, v in np.ndenumerate(a)), reverse=True)[:5]
print("five largest nudges:", "; ".join(f"{k}{list(i)} {G[k][i]:+.4f}" for _, k, i in big))
zero = [(k, i) for k, a in G.items() for i, v in np.ndenumerate(a) if v == 0.0]
print(f"nudges that are exactly zero: {len(zero)}:", ", ".join(f"{k}{list(i)}" for k, i in zero[:14]))

# the nudge on each seat's row
X = (tt.t["E"][torch.tensor(IDS)] + tt.t["P"][:6]).detach().requires_grad_(True)
S = tt.scores_from_rows(X[:-1])
Ls = -torch.log_softmax(S, -1).gather(-1, torch.tensor(IDS[1:]).unsqueeze(-1)).mean()
Ls.backward()
gX = X.grad.numpy()
print("\nthe nudge each seat's row receives (row entering layer 0), and its row:")
seat = ["the@0", "cat@1", "sat@2", "on@3", "the@4", "mat@5"]
for s in range(6):
    print(f"  {seat[s]:<6} row {r3(X.detach().numpy()[s])}  nudge {r3(gX[s], 4)}  length {np.linalg.norm(gX[s]):.4f}")
# the token table's row for 'the' = seat 0 + seat 4 + its use at the output
out_the = G["E"][0] - gX[0] - gX[4]
print("\nthe token table's row for 'the', id 0, collects three nudges:")
print(f"  from seat 0 {r3(gX[0], 4)} + from seat 4 {r3(gX[4], 4)} + from the output scores {r3(out_the, 4)} = {r3(G['E'][0], 4)}")
print(f"  autograd's nudge for E['the'] {r3(G['E'][0], 4)}; wiggled {r3(W['E'][0], 4)}")
print(f"  mat (id 4): seat 5 gives {r3(gX[5], 4)}, so its whole nudge {r3(G['E'][4], 4)} comes from the output (Section 17.2's rule {r3([-0.22601, -0.14605], 4)})")
print(f"  words that never appear (a, dog, rabbit, looked, at, hat, <unk>), output only:")
for w in range(5, 12):
    print(f"    {WORDS[w]:<7} {r3(G['E'][w], 4)}")
print("\nlayer 0's attention tables (two layers share nothing after this: each gets its own nudge):")
for k in ("h.0.att.W_q", "h.0.att.W_k", "h.0.att.W_v", "h.1.att.W_q"):
    print(f"  {k:<12} nudge {G[k].round(4).tolist()}")
CHECK = G["E"][1, 0]
print(f"\ncheckpoint: the nudge on cat's first number, E['cat'][0]: autograd {CHECK:+.6f}, wiggled {W['E'][1,0]:+.6f}")
save_json("head_nudges", dict(loss=L, max_diff=diffs[worst], seat_nudges=gX.tolist(), rows=X.detach().numpy().tolist(),
                              the=dict(seat0=gX[0].tolist(), seat4=gX[4].tolist(), out=out_the.tolist(), total=G["E"][0].tolist()),
                              E=G["E"].tolist(), check=float(CHECK), t_auto_ms=t_auto*1000, t_wig_ms=t_wig*1000,
                              diffs=diffs, n_zero=len(zero),
                              all_auto=[float(v) for k in G for v in G[k].ravel()], all_wig=[float(v) for k in G for v in W[k].ravel()]))
