"""Chapter 40, 40.2-40.4: linear probes on every layer's rows.  Windows of Chapter 24's pool (train: 200 windows of 128 tokens,
test: 100 other windows), rows at every seat from 1 to 127.  A probe is a multinomial logistic regression (one weight matrix and
bias, 120 steps of Adam at 0.05 on standardised rows, weight decay 1e-3).  Tasks: SPACE (does the current token begin with a space),
CAP (is the current token a capitalised word), NEXTSPACE (will the next token begin with a space), PREV (which of the 20 most
common tokens came one seat earlier, or none of them), CUR (which of the 20 most common tokens is this one, or none), POS (seat
in four buckets).  Control task for SPACE and NEXTSPACE (Hewitt and Liang 2019): each token id gets a fixed random label with the
same share of ones; selectivity = task accuracy minus control accuracy.  Models: trained Tiny, untrained Tiny (its starting values).
CPU, one thread, 32-bit, seed 1865."""
from lib import *
import numpy as np, sys
torch.set_num_threads(1); torch.manual_seed(SEED)
from _pool import corpora
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
T = tok(); c = corpora(); pool = np.array([int(i) for i in c["pool"]])
rng = np.random.default_rng(SEED); starts = rng.choice(len(pool) - 129, size=300, replace=False)
W = np.stack([pool[s:s + 128] for s in starts]); Wtr, Wte = W[:200], W[200:]
strs = [T.decode([i]) for i in range(2048)]
space = np.array([s.startswith(" ") for s in strs]); cap = np.array([s.strip()[:1].isupper() for s in strs])
cnt = np.bincount(pool, minlength=2048); common = list(np.argsort(-cnt)[:20]); cid = {int(t): i for i, t in enumerate(common)}
ctl = np.random.default_rng(SEED + 1).random(2048) < space.mean()      # control labels: random per token id, same share
def labels(Wn):
    cur = Wn[:, 1:127]; nxt = Wn[:, 2:128]; prv = Wn[:, 0:126]
    pos = np.broadcast_to(np.arange(1, 127), cur.shape)
    L = dict(SPACE=space[cur], CAP=cap[cur], NEXTSPACE=space[nxt], CTL_SPACE=ctl[cur], CTL_NEXTSPACE=ctl[nxt],
             PREV=np.vectorize(lambda t: cid.get(int(t), 20))(prv), CUR=np.vectorize(lambda t: cid.get(int(t), 20))(cur),
             POS=pos // 32)
    return {k: v.astype(int).ravel() for k, v in L.items()}
Ltr, Lte = labels(Wtr), labels(Wte)
def rows(tt, Wn):
    with torch.no_grad(): r = forward(tt, Wn)["resid"]
    return [x[:, 1:127].reshape(-1, x.shape[-1]) for x in r]            # 13 x (n*126, 128)
def fit(Xtr, ytr, Xte, yte, k):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    y = torch.tensor(ytr); w = torch.zeros(A.shape[1], k, requires_grad=True); b = torch.zeros(k, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05, weight_decay=1e-3)
    for _ in range(120):
        opt.zero_grad(); torch.nn.functional.cross_entropy(A @ w + b, y).backward(); opt.step()
    with torch.no_grad(): return float(((B @ w + b).argmax(-1).numpy() == yte).mean())
def majority(ytr, yte): return float((yte == np.bincount(ytr).argmax()).mean())
KS = dict(SPACE=2, CAP=2, NEXTSPACE=2, CTL_SPACE=2, CTL_NEXTSPACE=2, PREV=21, CUR=21, POS=4)
R = {}
for name in ["trained", "untrained"]:
    tt = load(CH24) if name == "trained" else TorchTiny.from_numpy(Tiny(), dtype=torch.float32)
    if name == "untrained":
        for k in tt.t: tt.t[k] = tt.t[k].float().detach()
    Xtr, Xte = rows(tt, Wtr), rows(tt, Wte); R[name] = {}
    for task, k in KS.items():
        R[name][task] = [fit(Xtr[l], Ltr[task], Xte[l], Lte[task], k) for l in range(13)]
        print(f"{name:9s} {task:13s} majority {majority(Ltr[task], Lte[task]):.3f}  by layer 0..12: " + " ".join(f"{a:.3f}" for a in R[name][task]), flush=True)
R["majority"] = {t: majority(Ltr[t], Lte[t]) for t in KS}; R["share_space"] = float(space.mean())
save_json("probe", R)
print("selectivity SPACE (trained):", " ".join(f"{a-b:+.3f}" for a, b in zip(R["trained"]["SPACE"], R["trained"]["CTL_SPACE"])))
print("selectivity SPACE (untrained):", " ".join(f"{a-b:+.3f}" for a, b in zip(R["untrained"]["SPACE"], R["untrained"]["CTL_SPACE"])))
print("selectivity NEXTSPACE (trained):", " ".join(f"{a-b:+.3f}" for a, b in zip(R["trained"]["NEXTSPACE"], R["trained"]["CTL_NEXTSPACE"])))
