"""Chapter 41, 41.4-41.5: a sparse autoencoder on Tiny's rows.  Rows entering block LAYER (default 6) of Chapter 24's Tiny, on 200 pool windows to train
(25,200 rows) and 100 others to score (12,600), rows divided by their average length.  Encoder: ReLU(x W_e + b_e) with 1,024 features (8 per row
width); decoder: unit-length columns; loss mean squared error + L1 * mean of the feature values (L1 = 3e-3 unless set); Adam 1e-3, batches of 256, 30 passes.
Reports variance explained, average active features (L0), dead features; the best single feature for 'the current token starts a word' against the
probe of Chapter 40 on the same rows; and for five features the tokens that make them largest.  CPU, one thread, 32-bit, seed 1865."""
from lib import *
import numpy as np, collections
torch.set_num_threads(1); torch.manual_seed(SEED)
from _pool import corpora
LAYER = int(os.environ.get("LAYER", "6")); L1 = float(os.environ.get("L1", "3e-3")); NF = int(os.environ.get("NF", "1024")); TAG = os.environ.get("TAG", "")
tt = load(CH24); T = tok(); c = corpora(); pool = np.array([int(i) for i in c["pool"]])
rng = np.random.default_rng(SEED); starts = rng.choice(len(pool) - 129, size=300, replace=False)   # the windows of ch40 probe.py
W = np.stack([pool[s:s + 128] for s in starts]); Wtr, Wte = W[:200], W[200:]
def rows(Wn):
    with torch.no_grad(): r = forward(tt, Wn)["resid"][LAYER]
    return r[:, 1:127].reshape(-1, 128), torch.tensor(Wn[:, 1:127].ravel())
Xtr, ttr = rows(Wtr); Xte, tte = rows(Wte); s = float(Xtr.norm(dim=-1).mean()); Xtr, Xte = Xtr / s, Xte / s
mu = Xtr.mean(0)
class SAE(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.We = torch.nn.Parameter(torch.randn(128, NF) * 0.05); self.be = torch.nn.Parameter(torch.zeros(NF))
        self.Wd = torch.nn.Parameter(torch.randn(NF, 128)); self.bd = torch.nn.Parameter(mu.clone())
    def dec(self): return self.Wd / self.Wd.norm(dim=1, keepdim=True)
    def forward(self, x):
        f = torch.relu(x @ self.We + self.be); return f @ self.dec() + self.bd, f
m = SAE(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
for ep in range(30):
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 256):
        x = Xtr[perm[i:i + 256]]; xh, f = m(x)
        loss = ((xh - x) ** 2).sum(-1).mean() + L1 * f.sum(-1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    if ep % 10 == 9 or ep == 0:
        with torch.no_grad(): xh, f = m(Xte); ve = 1 - float(((xh - Xte) ** 2).sum() / ((Xte - Xte.mean(0)) ** 2).sum())
        print(f"pass {ep + 1:2d}: held-out variance explained {ve:.3f}, mean active features {float((f > 0).float().sum(-1).mean()):.1f}", flush=True)
with torch.no_grad(): xh, f = m(Xte); ftr = m(Xtr)[1]
ve = 1 - float(((xh - Xte) ** 2).sum() / ((Xte - Xte.mean(0)) ** 2).sum()); l0 = float((f > 0).float().sum(-1).mean()); dead = int(((ftr > 0).sum(0) == 0).sum())
print(f"layer {LAYER}, {NF} features, L1 {L1}: held-out variance explained {ve:.3f}; mean active features per row {l0:.1f} of {NF}; dead features (never active on the training rows) {dead}")
strs = [T.decode([i]) for i in range(2048)]; space = torch.tensor([x.startswith(" ") for x in strs])
ltr = space[ttr].float(); lte = space[tte].float()
# best single feature for SPACE: threshold chosen on training rows
best = None
for j in range(NF):
    a = ftr[:, j]
    if (a > 0).sum() < 50: continue
    for th in (0.0, float(a[a > 0].median())):
        for sign in (1, -1):
            pred = ((a > th) if sign == 1 else (a <= th)).float(); acc = float((pred == ltr).float().mean())
            if best is None or acc > best[0]: best = (acc, j, th, sign)
acc, j, th, sign = best; pred = ((f[:, j] > th) if sign == 1 else (f[:, j] <= th)).float()
print(f"best single feature for 'the current token begins a word': feature {j}, threshold {th:.3f}, {'on' if sign == 1 else 'off'} = word-start; train {acc:.3f}, held-out {float((pred == lte).float().mean()):.3f} (majority {float(max(lte.mean(), 1 - lte.mean())):.3f})")
# five most frequent features: tokens at their top activations
order = torch.argsort(-(ftr > 0).float().sum(0))
R = dict(layer=LAYER, nf=NF, l1=L1, ve=ve, l0=l0, dead=dead, single=[float(acc), float((pred == lte).float().mean())])
shown = []
mx = f.max(0).values; cand = [int(k) for k in torch.argsort(-mx)[:400] if (ftr[:, k] > 0).sum() > 100]
for k in cand[:8]:
    top = torch.topk(f[:, k], 40).indices; cnt = collections.Counter(strs[int(tte[i])] for i in top)
    nact = int((f[:, k] > 0).sum()); print(f"feature {k:4d} active on {nact:5d} of {len(f)} rows; its 40 largest rows are tokens: " + ", ".join(f"{t!r} x{n}" for t, n in cnt.most_common(5)))
    shown.append((k, nact, cnt.most_common(5)))
R["shown"] = [[k, n, [[t, c] for t, c in cc]] for k, n, cc in shown]
save_json(f"sae{TAG}", R)
pass  # (the dictionary is not saved: it is rebuilt in a minute by this script)
