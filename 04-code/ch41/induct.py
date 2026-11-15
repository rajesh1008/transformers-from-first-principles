"""Chapter 41, 41.5: an induction head found by intervention.  Tiny continued for 1,500 steps on windows that repeat a 64-token passage
(grow_induction.py, lr 1e-3; the Chapter 36 recipe; checkpoint in outputs/ckpt/, not in the code zip).  Test material: 20 runs of 60 random
tokens (ids 256 to 2047) shown twice.  For every head: induction score (mean share, from each seat of the second run, on the seat AFTER the
earlier occurrence of the current token), previous-token score (share on the seat before), and the rise in loss on the second run when the head's
output is zeroed before W_o.  Then the circuit test: the loss on the second run with the top induction head zeroed, the top previous-token head
zeroed, both, and ten random single heads for comparison; and the induction score of the top head when the previous-token head is zeroed.
CPU, one thread, 32-bit, seed 1865."""
from lib import *
import numpy as np
torch.set_num_threads(1)
tt = load(CH24); sd = torch.load(os.path.join(HERE, "outputs", "ckpt", "repeat_1e-3_1500.pt"))
for k in tt.t: tt.t[k] = sd[k].float().detach()
rng = np.random.default_rng(SEED); N = 60; NL, H = 12, 4
X = torch.tensor(np.stack([np.concatenate([r, r]) for r in (rng.integers(256, 2048, size=N) for _ in range(20))]))
def second_loss(**kw):
    with torch.no_grad(): lg = forward(tt, X[:, :-1], **kw)["logits"]
    L = torch.nn.functional.cross_entropy(lg.reshape(-1, 2048), X[:, 1:].reshape(-1), reduction="none").reshape(len(X), -1)
    return float(L[:, :N - 1].mean()), float(L[:, N - 1:].mean())
def scores(**kw):
    with torch.no_grad(): A = forward(tt, X[:, :-1], want_att=True, **kw)["att"]
    q = torch.arange(N, 2 * N - 1); ind = np.zeros((NL, H)); prev = np.zeros((NL, H))
    for l in range(NL):
        ind[l] = A[l][:, :, q, q - N + 1].sum(-1).mean(0).numpy() / (N - 1); prev[l] = A[l][:, :, q, q - 1].sum(-1).mean(0).numpy() / (N - 1)
    return ind, prev
f0, s0 = second_loss(); ind, prev = scores()
print(f"loss on the first run {f0:.3f} nats, on the repeat {s0:.3f} (uniform guess 7.625)")
d = np.zeros((NL, H))
for l in range(NL):
    for h in range(H): d[l, h] = second_loss(ablate={(l, h)})[1] - s0
fl = lambda i: f"{i // H}.{i % H}"
print("top induction scores: " + ", ".join(f"{fl(i)} {ind.ravel()[i]:.3f}" for i in np.argsort(-ind.ravel())[:5]))
print("top previous-token scores: " + ", ".join(f"{fl(i)} {prev.ravel()[i]:.3f}" for i in np.argsort(-prev.ravel())[:5]))
print("largest rises in repeat loss when one head is zeroed: " + ", ".join(f"{fl(i)} {d.ravel()[i]:+.3f}" for i in np.argsort(-d.ravel())[:6]))
from scipy.stats import spearmanr
print(f"Spearman between induction score and repeat-loss rise over 48 heads: {spearmanr(ind.ravel(), d.ravel()).statistic:.3f}; between previous-token score and rise: {spearmanr(prev.ravel(), d.ravel()).statistic:.3f}")
hi = int(np.argmax(ind.ravel())); hp = int(np.argmax(prev.ravel())); HI = (hi // H, hi % H); HP = (hp // H, hp % H)
print(f"top induction head {fl(hi)}, top previous-token head {fl(hp)}")
for nm, S in (("nothing removed", set()), (f"induction head {fl(hi)}", {HI}), (f"previous-token head {fl(hp)}", {HP}), ("both", {HI, HP})):
    print(f"   repeat loss with {nm:28s}: {second_loss(ablate=S)[1]:.3f}")
rr = np.random.default_rng(SEED + 3); rl = []
for _ in range(10):
    i = int(rr.integers(NL * H)); rl.append(second_loss(ablate={(i // H, i % H)})[1])
print(f"   repeat loss with one random head removed, ten draws: mean {np.mean(rl):.3f}, range {min(rl):.3f} to {max(rl):.3f}")
ind2, _ = scores(ablate={HP}); print(f"   induction score of head {fl(hi)}: {ind[HI]:.3f}; with previous-token head {fl(hp)} zeroed: {ind2[HI]:.3f}")
for k in (1, 2, 3, 5):
    ti = {(i // H, i % H) for i in np.argsort(-ind.ravel())[:k]}; ta = {(i // H, i % H) for i in np.argsort(-d.ravel())[:k]}
    print(f"   zero the top {k} by induction score: repeat loss {second_loss(ablate=ti)[1]:.3f}; top {k} by ablation: {second_loss(ablate=ta)[1]:.3f}")
save_json("induct", dict(first=f0, repeat=s0, ind=ind.tolist(), prev=prev.tolist(), drop=d.tolist()))
