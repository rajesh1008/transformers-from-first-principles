"""Lab 42: exact checks on the mixture-of-experts layer before any trained number is read.  CPU, one thread.  python3 lab42.py"""
import os, importlib, sys, math
os.environ.update(KIND="moe", E="1", K="1", HID="64", NL="2")
import torch
torch.set_grad_enabled(False)
ok = 0; n = 0
def check(name, cond):
    global ok, n; n += 1; ok += bool(cond); print(("[pass] " if cond else "[FAIL] ") + name)
import moe as M1
torch.manual_seed(1865)
x = torch.randn(2, 5, 128)
# 1 one expert, top-1: the MoE layer is exactly the dense layer with the same weights (the gate is a softmax over one score, so 1)
lay = M1.MoE(); dense = M1.Expert(64); dense.load_state_dict(lay.experts[0].state_dict())
d = float((lay(x) - dense(x)).abs().max()); print(f"  largest difference, one-expert MoE against dense: {d}")
check("a one-expert MoE is the dense feed-forward layer", d == 0.0)
check("with one expert the balance term is exactly 1", abs(float(lay.aux) - 1.0) < 1e-6)
# reload with 4 experts, top-2
os.environ.update(E="4", K="2"); M = importlib.reload(M1)
lay = M.MoE(); y = lay(x); p = lay.last["p"]; idx = lay.last["idx"]
g = p.gather(-1, idx); g = g / g.sum(-1, keepdim=True)
check("top-2 gates, renormalised, sum to 1 for every token", float((g.sum(-1) - 1).abs().max()) < 1e-6)
# hand-built output: the sum over the two chosen experts of gate x expert output
xf = x.reshape(-1, 128); ref = torch.zeros_like(xf)
for t in range(xf.shape[0]):
    for j in range(2): ref[t] += g[t, j] * lay.experts[int(idx[t, j])](xf[t:t + 1])[0]
d = float((y.reshape(-1, 128) - ref).abs().max()); print(f"  largest difference against a token-by-token loop: {d:.2e}")
check("the batched MoE equals a token-by-token loop", d < 1e-6)
# balance term: uniform router shares and an even split give exactly 1
lay.router.weight.data.zero_(); lay(x); print(f"  balance term with a zero router: {float(lay.aux):.4f}")
check("a zero router (uniform shares) gives a balance term of 1", abs(float(lay.aux) - 1.0) < 1e-6)
# capacity: a generous limit drops nothing, a tight one drops tokens
lay = M.MoE(); lay.cap = 100.0; y2 = lay(x); check("a capacity factor of 100 drops no token", lay.last["dropped"] == 0)
lay.cap = 0.25; lay(x); print(f"  slots dropped at capacity factor 0.25: {lay.last['dropped']} of {lay.last['n'] * 2}")
check("a capacity factor of 0.25 drops at least half of the slots", lay.last["dropped"] >= lay.last["n"])
# removing an expert removes exactly its contribution
lay = M.MoE(); y = lay(x).reshape(-1, 128); idx = lay.last["idx"]; p = lay.last["p"]; g = p.gather(-1, idx); g = g / g.sum(-1, keepdim=True)
lay.drop = 0; y0 = lay(x).reshape(-1, 128); contrib = torch.zeros_like(y)
for t in range(y.shape[0]):
    for j in range(2):
        if int(idx[t, j]) == 0: contrib[t] += g[t, j] * lay.experts[0](xf[t:t + 1])[0]
check("dropping expert 0 removes exactly expert 0's weighted output", float((y - y0 - contrib).abs().max()) < 1e-6)
# parameter count: total minus (E - K) unused experts per layer
net = M.Net(); tot, act = M.count(net); per = sum(q.numel() for q in net.h[0].ff.experts[0].parameters())
check("used-per-token count is total minus (E - K) experts per layer", tot - act == 2 * (4 - 2) * per)
print(f"{ok} of {n} checks pass")
