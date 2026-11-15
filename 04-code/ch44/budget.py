"""Chapter 44: the scratchpad as a budget, and voting as test-time compute, on trained add.py models.
NAME (a pad model): for each budget k, the model writes at most k scratchpad characters and is then made to write '>' and
answer (k = 0 is no scratchpad at all; the breakage); exact match on the held-out pairs.
VOTE (a direct or pad model): sample N answers at temperature TEMP (1.0) and take the commonest; N = 1, 5, 15.
Writes outputs/budget_NAME.json."""
import os, json, numpy as np, torch
NAME = os.environ.get("NAME", "pad_s1865"); HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(HERE, "outputs", NAME + ".json")))
os.environ.update(FORMAT=cfg["format"], DIGITS=str(cfg.get("digits", 2)))
import add
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
m = add.Net(); m.load_state_dict(torch.load(os.path.join(HERE, "outputs", "ckpt", NAME + ".pt"))); m.eval()
tr, te = add.pairs(); te = te[:1000] if len(te) > 1000 else te
Q = [f"{a}+{b}=" for a, b in te]; S = [a + b for a, b in te]; R = dict(name=NAME, format=cfg["format"], n=len(te), budget=[], vote=[])
L = [len(add.text(a, b)[1].split(">")[0]) for a, b in te]; full = max(L)
if cfg["format"] == "pad":
    ks = sorted(set([0, 2, 4, 6, 9, 14, full] if add.DIG == 2 else [0, 4, 9, 18, 27, 32, full])) + [None]
    print(f"scratchpad lengths on these pairs: {min(L)} to {max(L)} characters", flush=True)
    for k in ks:
        g = add.generate(m, Q, force=k); acc = float(np.mean([add.answer(x) == s for x, s in zip(g, S)]))
        R["budget"].append(dict(k=k, exact=acc, example=g[0])); print(f"scratchpad cut after {k if k is not None else 'no'} characters: exact {acc:.4f}   e.g. {Q[0]}{g[0]}", flush=True)
TEMP = float(os.environ.get("TEMP", "1.0")); gen = torch.Generator().manual_seed(1865)
for N in (1, 5, 15):
    votes = [[add.answer(x) for x in add.generate(m, Q, temp=TEMP, rng=gen)] for _ in range(N)]
    pick = []
    for i in range(len(Q)):
        c = {}; [c.__setitem__(v[i], c.get(v[i], 0) + 1) for v in votes if v[i] is not None]
        pick.append(max(c, key=c.get) if c else None)
    acc = float(np.mean([p == s for p, s in zip(pick, S)])); one = float(np.mean([v == s for v, s in zip(votes[0], S)]))
    R["vote"].append(dict(N=N, exact=acc)); print(f"sampled at temperature {TEMP}, majority of {N:2d}: exact {acc:.4f}", flush=True)
json.dump(R, open(os.path.join(HERE, "outputs", f"budget_{NAME}.json"), "w"), indent=1)
