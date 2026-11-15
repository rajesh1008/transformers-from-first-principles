"""Chapter 44: give a small transformer a scratchpad.  Two-digit addition (a + b, a and b from 10 to 99), written as characters.
DIGITS=4 (tick 33): four-digit numbers, 20,000 training and 2,000 held-out pairs, scratchpad "d+d+c=s;" for every column.
  FORMAT=direct   "37+48=>85."                       the answer straight after the question
  FORMAT=pad      "37+48=7+8=15;3+4+1=8;>85."        ones column, then tens column with the carry, then the answer
A four-block decoder (width 128, 4 heads, pre-norm, GELU, learned positions, causal mask), trained from scratch on the
characters after '=' (the question is given), AdamW 1e-3 with 100 warm-up steps and cosine decay, batch 128, STEPS steps.
The 8,100 (a, b) pairs are split once (seed 1865): 6,480 for training, 1,620 held out.  Test = greedy decoding on the held-out
pairs, exact match of the answer between '>' and '.'.  CPU, THREADS threads (2 for the book runs), 32-bit, seed SEED.  Writes outputs/NAME.json, ckpt/NAME.pt."""
import os, json, math, time, numpy as np, torch
torch.set_num_threads(int(os.environ.get("THREADS", "1")))   # THREADS: 2 for the tick-33 runs
HERE = os.path.dirname(os.path.abspath(__file__)); env = os.environ.get
NAME = env("NAME", "direct"); FORMAT = env("FORMAT", "direct"); SEED = int(env("SEED", "1865")); STEPS = int(env("STEPS", "2000")); DIG = int(env("DIGITS", "2"))
D, NL, NH = 128, 4, 4; T = 32 if DIG == 2 else 16 * DIG
CH = list("0123456789+=;>.") + ["<pad>"]; ci = {c: i for i, c in enumerate(CH)}; PAD = ci["<pad>"]
def text(a, b, fmt=FORMAT):
    q = f"{a}+{b}="
    if fmt == "direct": return q, f">{a+b}."
    if DIG == 2:
        o = a % 10 + b % 10; c = o // 10; t = a // 10 + b // 10 + c
        return q, f"{a%10}+{b%10}={o};{a//10}+{b//10}+{c}={t};>{a+b}."
    pad, c, x, y = "", 0, a, b                       # DIGITS > 2: one column at a time from the ones, carry written in every column
    for _ in range(DIG):
        o = x % 10 + y % 10 + c; pad += f"{x%10}+{y%10}+{c}={o};"; c = o // 10; x //= 10; y //= 10
    return q, pad + f">{a+b}."
def pairs():
    if DIG > 2:                                      # 20,000 training and 2,000 held-out pairs, disjoint, seed 1865
        g = np.random.default_rng(1865); lo, hi = 10 ** (DIG - 1), 10 ** DIG; S = set(); L = []
        while len(L) < 22000:
            p = (int(g.integers(lo, hi)), int(g.integers(lo, hi)))
            if p not in S: S.add(p); L.append(p)
        return L[:20000], L[20000:]
    P = [(a, b) for a in range(10, 100) for b in range(10, 100)]; idx = np.random.default_rng(1865).permutation(len(P))
    return [P[i] for i in idx[:6480]], [P[i] for i in idx[6480:]]
class Block(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.ln1 = torch.nn.LayerNorm(D); s.ln2 = torch.nn.LayerNorm(D); s.qkv = torch.nn.Linear(D, 3 * D); s.o = torch.nn.Linear(D, D)
        s.ff = torch.nn.Sequential(torch.nn.Linear(D, 4 * D), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(4 * D, D))
    def forward(s, x):
        B, n, _ = x.shape; q, k, v = s.qkv(s.ln1(x)).split(D, -1); sp = lambda t: t.view(B, n, NH, D // NH).transpose(1, 2)
        a = torch.nn.functional.scaled_dot_product_attention(sp(q), sp(k), sp(v), is_causal=True)
        x = x + s.o(a.transpose(1, 2).reshape(B, n, D)); return x + s.ff(s.ln2(x))
class Net(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.E = torch.nn.Embedding(len(CH), D); s.P = torch.nn.Parameter(torch.randn(1, T, D) * 0.02)
        s.h = torch.nn.ModuleList([Block() for _ in range(NL)]); s.lnf = torch.nn.LayerNorm(D); s.out = torch.nn.Linear(D, len(CH))
    def forward(s, ids):
        x = s.E(ids) + s.P[:, :ids.shape[1]]
        for b in s.h: x = b(x)
        return s.out(s.lnf(x))
def batch(P, rng, n=128):
    sel = [P[i] for i in rng.integers(0, len(P), n)]; X = torch.full((n, T), PAD); M = torch.zeros(n, T, dtype=torch.bool)
    for r, (a, b) in enumerate(sel):
        q, ans = text(a, b); s = [ci[c] for c in q + ans]; X[r, :len(s)] = torch.tensor(s); M[r, len(q):len(s)] = True
    return X, M
@torch.no_grad()
def generate(m, prompts, stop=".", maxlen=T, temp=0.0, rng=None, force=None):
    """Greedy (temp 0) or sampled continuation of a batch of prompts of equal length, until `stop` or maxlen.
    force=k: after k generated characters, if no '>' has been written yet, write '>' (the truncation breakage)."""
    S = torch.tensor([[ci[c] for c in p] for p in prompts]); n0 = S.shape[1]; done = torch.zeros(len(prompts), dtype=torch.bool)
    seen = torch.zeros(len(prompts), dtype=torch.bool)
    for k in range(maxlen - n0):
        lg = m(S)[:, -1]
        c = lg.argmax(-1) if temp == 0 else torch.multinomial(torch.softmax(lg / temp, -1), 1, generator=rng)[:, 0]
        if force is not None and k == force: c = torch.where(seen, c, torch.full_like(c, ci[">"]))
        c = torch.where(done, torch.full_like(c, PAD), c); seen |= c == ci[">"]; done |= c == ci[stop]
        S = torch.cat([S, c[:, None]], 1)
        if done.all(): break
    return ["".join(CH[i] for i in r[n0:].tolist() if i != PAD) for r in S]
def answer(gen):
    if ">" not in gen or "." not in gen.split(">", 1)[1]: return None
    a = gen.split(">", 1)[1].split(".", 1)[0]
    return int(a) if a.isdigit() else None
if __name__ == "__main__":
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED); tr, te = pairs(); m = Net()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
    lr = lambda s: 1e-3 * (s / 100 if s < 100 else 0.5 * (1 + math.cos(math.pi * (s - 100) / (STEPS - 100))))
    R = dict(name=NAME, format=FORMAT, seed=SEED, steps=STEPS, params=sum(p.numel() for p in m.parameters()), log=[]); t0 = time.perf_counter()
    q0, a0 = text(37, 48); print(f"{NAME}: format {FORMAT}, e.g. '{q0}{a0}' ({len(a0)} characters after '='); learned numbers {R['params']:,}", flush=True)
    for step in range(1, STEPS + 1):
        for g in opt.param_groups: g["lr"] = lr(step)
        X, M = batch(tr, rng); lg = m(X[:, :-1]); y = X[:, 1:].clone(); y[~M[:, 1:]] = -100
        L = torch.nn.functional.cross_entropy(lg.reshape(-1, len(CH)), y.reshape(-1), ignore_index=-100); opt.zero_grad(); L.backward(); opt.step()
        if step % 500 == 0 or step == STEPS:
            m.eval(); sub = te[:200]; g = generate(m, [f"{a}+{b}=" for a, b in sub]); acc = float(np.mean([answer(x) == a + b for x, (a, b) in zip(g, sub)])); m.train()
            R["log"].append(dict(step=step, loss=L.item(), held200=acc)); print(f"step {step:5d}  loss {L.item():.4f}  exact on 200 held-out pairs {acc:.3f}  ({time.perf_counter()-t0:.0f} s)", flush=True)
    m.eval(); g = generate(m, [f"{a}+{b}=" for a, b in te]); R["test"] = float(np.mean([answer(x) == a + b for x, (a, b) in zip(g, te)]))
    R["digits"] = DIG; R["carry"] = float(np.mean([answer(x) == a + b for x, (a, b) in zip(g, te) if a % 10 + b % 10 >= 10])); R["nocarry"] = float(np.mean([answer(x) == a + b for x, (a, b) in zip(g, te) if a % 10 + b % 10 < 10]))
    R["example"] = g[[i for i, (a, b) in enumerate(te)][0]]; R["example_q"] = f"{te[0][0]}+{te[0][1]}="
    print(f"held-out exact {R['test']:.4f} ({len(te):,} pairs; ones column carries {R['carry']:.4f}, without {R['nocarry']:.4f}); e.g. {R['example_q']}{R['example']}", flush=True)
    json.dump(R, open(os.path.join(HERE, "outputs", NAME + ".json"), "w"), indent=1)
    os.makedirs(os.path.join(HERE, "outputs", "ckpt"), exist_ok=True); torch.save(m.state_dict(), os.path.join(HERE, "outputs", "ckpt", NAME + ".pt"))
