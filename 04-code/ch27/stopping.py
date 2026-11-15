"""Chapter 27, section 27.9 follow-up: why experiments.py Part I almost never stopped.  Tiny writes a blank line as two
single line-break tokens far more often than as the merged token, so a stop rule on token ids misses it; a stop rule on
the decoded text catches it.  Also the tail of a 200-token greedy continuation (27.6)."""
from _common import *
import torch
torch.set_num_threads(1)
from tiny.eval import load
from tiny.seed import streams
from tiny import generate as G

tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
t = tok()
P = t.encode("the cat sat on the")
STOP = [i for i in range(2048) if "\n\n" in t.decode([i])]
NL = t.encode("\n")
print(f"tokens containing a blank line: {[(i, t.decode([i])) for i in STOP]}; a single line break is {NL}")
ids, _ = G.generate(tt, P, 200, None)
new = ids[len(P):]
pairs = sum(1 for a, b in zip(new, new[1:]) if [a] == NL and [b] == NL)
print(f"greedy 200 tokens: merged blank-line tokens {sum(x in STOP for x in new)}; two single line breaks in a row {pairs}")
print(f"first blank line in the decoded text at character {t.decode(new).find(chr(10) * 2)}")
print(f"greedy tail (last 40 tokens): {t.decode(new[-40:])!r}")
print(f"the pool tokenizes a blank line as {t.encode(chr(10) * 2)}")
res = {}
print("\nStop when the decoded continuation contains a blank line, cap 200, 30 samples per rule, seed 1865")
for label, kw in [("T 0.7", dict(T=0.7)), ("T 1.0", dict(T=1.0)), ("T 1.3", dict(T=1.3)), ("min-p 0.1", dict(min_p=0.1)),
                  ("T 1.3 + min-p 0.1", dict(T=1.3, min_p=0.1))]:
    rng = streams(SEED)["sample"]
    L, capped = [], 0
    for _ in range(30):
        cur = list(P)
        for step in range(200):
            q, _ = G.shares(G.next_scores(tt, cur), **kw)
            cur.append(int(rng.choice(len(q), p=q)))
            if "\n\n" in t.decode(cur[len(P):]):
                break
        else:
            capped += 1
        L.append(len(cur) - len(P))
    res[label] = dict(median=float(np.median(L)), mean=float(np.mean(L)), capped=capped, lengths=L)
    print(f"  {label:<18} median length {np.median(L):5.0f}  mean {np.mean(L):6.1f}  shortest {min(L):3d}  longest {max(L):3d}  reached the cap {capped}/30")
save_json("stopping", res)
