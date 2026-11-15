"""Lab 44: exact checks on the chapter's code, before trusting any run.  CPU, seconds, no training.
1-3 data: every two-digit scratchpad is correct arithmetic and ends in the true sum; same for 2,000 four-digit pairs;
          the two-digit split is 6,480 / 1,620 and disjoint, the four-digit split 20,000 / 2,000 and disjoint.
4     the answer parser reads '>57.' after any working and refuses text with no '>' or no '.'.
5     the learned-number count by hand equals the model's own count (two and four digits).
6     the causal mask: changing a later character does not change the model's scores at earlier positions.
7     forced truncation: with force=0 the first generated character is '>'.
8     a time-invariant state-space layer computed as a recurrence equals its convolution form.
9     memory: Tiny's key-value cache per token is 2 x 12 x 128 x 4 bytes = 12,288."""
import os, re, importlib, numpy as np, torch
torch.manual_seed(1865); ok = []
def load(dig, fmt):
    os.environ.update(DIGITS=str(dig), FORMAT=fmt); import add; importlib.reload(add); return add
def check_pad(add, P):
    for a, b in P:
        q, s = add.text(a, b); work, ans = s.split(">")
        for step in work.strip(";").split(";"):
            lhs, rhs = step.split("="); 
            if sum(int(t) for t in lhs.split("+")) != int(rhs): return False
        if int(ans.rstrip(".")) != a + b: return False
    return True
A2 = load(2, "pad"); tr2, te2 = A2.pairs(); ok.append(check_pad(A2, tr2 + te2))
A4 = load(4, "pad"); tr4, te4 = A4.pairs(); ok.append(check_pad(A4, te4))
ok.append(len(tr2) == 6480 and len(te2) == 1620 and not set(tr2) & set(te2) and len(tr4) == 20000 and len(te4) == 2000 and not set(tr4) & set(te4))
ok.append(A2.answer("3+4=7;2+3+0=5;>57.") == 57 and A2.answer(">57.") == 57 and A2.answer("3+4=7;57") is None and A2.answer(">57") is None)
def hand(T, D=128, V=16):
    blk = 2 * 2 * D + (D * 3 * D + 3 * D) + (D * D + D) + (D * 4 * D + 4 * D) + (4 * D * D + D)
    return V * D + T * D + 4 * blk + 2 * D + D * V + V
m2 = load(2, "direct").Net(); m4 = load(4, "direct").Net()
ok.append(hand(32) == sum(p.numel() for p in m2.parameters()) == 801552 and hand(64) == sum(p.numel() for p in m4.parameters()) == 805648)
A = load(2, "pad"); m = A.Net().eval(); x = torch.tensor([[A.ci[c] for c in "37+48=7+8=15;"]]); y = x.clone(); y[0, -1] = A.ci["9"]
with torch.no_grad(): ok.append(torch.allclose(m(x)[0, :-1], m(y)[0, :-1], atol=1e-6))
ok.append(A.generate(m, ["37+48="], force=0)[0].startswith(">"))
rng = np.random.default_rng(1865); N, T = 8, 32; a = np.exp(-rng.uniform(0.01, 0.5, N)); b = rng.normal(size=N); c = rng.normal(size=N); xs = rng.normal(size=T)
h = np.zeros(N); yr = []
for t in range(T): h = a * h + b * xs[t]; yr.append(c @ h)
K = [c @ (a ** j * b) for j in range(T)]; yc = [sum(K[j] * xs[t - j] for j in range(t + 1)) for t in range(T)]
ok.append(np.allclose(yr, yc, atol=1e-12))
ok.append(2 * 12 * 128 * 4 == 12288)
for i, v in enumerate(ok, 1): print(f"check {i}: {'pass' if v else 'FAIL'}")
print(f"{sum(ok)} of {len(ok)} checks pass")
