"""Lab 41: checks for Chapter 41.  Cheap exact checks on the instruments of lens.py, induct.py and mean_abl.py, plus the saved outputs.
No model call.  CPU, one thread, 32-bit, seed 1865."""
from lib import *
import numpy as np
torch.set_num_threads(1)
ok = 0; n = 0
def check(msg, cond):
    global ok, n
    n += 1; ok += bool(cond); print(("[pass] " if cond else "[FAIL] ") + msg)
tt = load(CH24); ids = torch.tensor([[5, 77, 300, 12, 900, 14, 15, 1600, 42, 7]])
o = forward(tt, ids)
check("the recording forward pass equals the book's forward pass", float((o["logits"] - tt.scores(ids)).abs().max()) == 0.0)
lens = tt._norm(o["resid"][12], "ln_f") @ tt.t["E"].T
d = float((lens - o["logits"]).abs().max()); print(f"  logit lens at layer 12 against the model's scores: largest difference {d:.1e}")
check("the logit lens at layer 12 is the model's own output", d == 0.0)
p = torch.log_softmax(o["logits"], -1)
kl = float((p.exp() * (p - p)).sum(-1).mean()); print(f"  KL of the final shares to themselves: {kl}")
check("KL of a distribution to itself is zero", kl == 0.0)
q = torch.log_softmax(torch.roll(o["logits"], 1, -1), -1)
k1 = float((p.exp() * (p - q)).sum(-1).mean()); k2 = float((q.exp() * (q - p)).sum(-1).mean()); print(f"  KL one way {k1:.3f}, the other way {k2:.3f}")
check("KL is not symmetric and is not negative", k1 > 0 and k2 > 0 and abs(k1 - k2) > 1e-6)
z = {(l, h): torch.zeros(32) for l in range(12) for h in range(4)}
a = forward(tt, ids, ablate={(3, 1)})["logits"]; b = forward(tt, ids, ablate={(3, 1)}, meanz=z)["logits"]
check("mean replacement with a mean of zero is zeroing", float((a - b).abs().max()) == 0.0)
grab = {}; forward(tt, ids, grab=grab)
check("grab returns each head's output before W_o with shape (batch, heads, seats, 32)", tuple(grab[3].shape) == (1, 4, 10, 32))
N = 6; A = torch.zeros(2 * N, 2 * N)
for t in range(N, 2 * N): A[t, t - N + 1] = 1.0
q = torch.arange(N, 2 * N - 1); sc = float(A[q, q - N + 1].sum() / (N - 1))
check("the induction score of a perfect induction pattern is 1", sc == 1.0)
A2 = torch.full((2 * N, 2 * N), 1.0 / (2 * N)); sc2 = float(A2[q, q - N + 1].sum() / (N - 1))
print(f"  even attention scores {sc2:.4f}")
check("a flat attention matrix over 2N seats scores 1/(2N) on the induction seat", abs(sc2 - 1 / (2 * N)) < 1e-6)
sd = torch.load(os.path.join(HERE, "outputs", "ckpt", "repeat_1e-3_1500.pt"))
diff = sum(float((sd[k].float() - tt.t[k]).abs().sum()) for k in tt.t); print(f"  summed absolute change of every weight after the repeat run: {diff:.1f}")
check("the grown copy differs from Chapter 24's Tiny and has the same shapes", diff > 0 and all(sd[k].shape == tt.t[k].shape for k in tt.t))
R = load_json("induct"); ind = np.array(R["ind"]); d2 = np.array(R["drop"])
check("the saved induction table puts head 10.0 first and the saved ablation table puts it first", int(ind.argmax()) == 40 and int(d2.argmax()) == 40)
M = load_json("mean_abl"); check("the saved mean-replacement run has the same intact loss as induct.py", abs(M["base"] - R["repeat"]) < 1e-6)
print(f"\n{ok} of {n} checks pass"); assert ok == n
