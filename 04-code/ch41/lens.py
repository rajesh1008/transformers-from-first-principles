"""Chapter 41, 41.1-41.3: the logit lens and a tuned lens on Tiny.  The logit lens reads the rows entering block l through the model's own last norm and
tied table, as if the model stopped there.  The tuned lens first passes the rows through a learned affine map per layer (128 x 128 and a bias, started at
the identity) trained to match the final next-token distribution (KL, Adam 3e-3, 150 steps full batch on 5,000 rows drawn from 200 windows) and is scored on 100 other windows.
Windows of Chapter 24's pool, 128 tokens, seats 1 to 126.  Then the logit lens on the 30 taught facts (Chapter 38's fine-tuned Tiny): the first layer at
which the taught verb is the lens's first choice at the answer's animal name.  CPU, one thread, 32-bit, seed 1865."""
from lib import *
import numpy as np
torch.set_num_threads(1); torch.manual_seed(SEED)
from _pool import corpora
tt = load(CH24); c = corpora(); pool = np.array([int(i) for i in c["pool"]])
rng = np.random.default_rng(SEED + 11); starts = rng.choice(len(pool) - 129, size=300, replace=False)
W = torch.tensor(np.stack([pool[s:s + 129] for s in starts])); Wtr, Wte = W[:200], W[200:]
def lens_logits(model, r): return model._norm(r, "ln_f") @ model.t["E"].T
with torch.no_grad():
    o_tr = forward(tt, Wtr[:, :128]); o_te = forward(tt, Wte[:, :128])
fin_tr = torch.log_softmax(o_tr["logits"], -1); fin_te = torch.log_softmax(o_te["logits"], -1)
def metrics(lp, fin, Y):
    p = lp.exp(); nll = -lp.gather(-1, Y.unsqueeze(-1)).squeeze(-1).mean()
    kl = (fin.exp() * (fin - lp)).sum(-1).mean(); top1 = (lp.argmax(-1) == Y).float().mean(); agree = (lp.argmax(-1) == fin.argmax(-1)).float().mean()
    return float(nll), float(kl), float(top1), float(agree)
Y = Wte[:, 1:]; R = dict(logit=[], tuned=[])
flat = torch.arange(200 * 128); idx = torch.tensor(np.random.default_rng(SEED).choice(200 * 128, size=5000, replace=False))
fin_tr = fin_tr.reshape(-1, 2048); o_tr = dict(resid=[r.reshape(-1, 128) for r in o_tr['resid']])
print("layer  logit lens: loss  KL to final  true top-1  agrees with final | tuned lens: loss  KL  true top-1  agrees")
Ts = []
for l in range(13):
    with torch.no_grad(): lp = torch.log_softmax(lens_logits(tt, o_te["resid"][l]), -1)
    m1 = metrics(lp, fin_te, Y); R["logit"].append(m1)
    if l == 12: m2 = m1
    else:
        A = torch.eye(128, requires_grad=True); b = torch.zeros(128, requires_grad=True); opt = torch.optim.Adam([A, b], lr=3e-3)
        xtr = o_tr["resid"][l].detach()[idx]; xte = o_te["resid"][l].detach(); ft_ = fin_tr[idx]
        for _ in range(150):
            opt.zero_grad(); lpt = torch.log_softmax(lens_logits(tt, xtr @ A + b), -1)
            kl = (ft_.exp() * (ft_ - lpt)).sum(-1).mean(); kl.backward(); opt.step()
        with torch.no_grad(): m2 = metrics(torch.log_softmax(lens_logits(tt, xte @ A + b), -1), fin_te, Y)
    R["tuned"].append(m2)
    print(f"{l:3d}     {m1[0]:8.3f} {m1[1]:10.3f} {m1[2]:10.3f} {m1[3]:12.3f}   |  {m2[0]:8.3f} {m2[1]:7.3f} {m2[2]:9.3f} {m2[3]:9.3f}", flush=True)
print(f"final model: loss {metrics(fin_te, fin_te, Y)[0]:.3f}, top-1 {metrics(fin_te, fin_te, Y)[2]:.3f}; uniform guess 7.625")
# the taught facts
save_json("lens", R)
