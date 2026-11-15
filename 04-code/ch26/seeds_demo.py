"""Chapter 26, sections 26.2 and 26.3: what a seed seeds, and where randomness enters Tiny.
  1  Tiny(seed=1865) built twice: same fingerprint.  torch.manual_seed changed between the two builds: still the same
     (Tiny's starting tables come from NumPy generators, not from PyTorch's).  Tiny(seed=7): different.
  2  The reading order: ShardStream(seed) first batch for seeds 1865, 1865, 7.
  3  One shared generator against one generator per job: dropout masks drawn after the starting tables, with and
     without one extra draw added to the start (a new table, say).  With a shared generator every mask moves;
     with tiny/seed.py's streams() the masks stay put.
  4  PyTorch's own generator: torch.manual_seed(1865) then a dropout mask, twice; and the same after one extra draw.
  5  Python's string hashing: the order of a set of words in two fresh processes, with and without PYTHONHASHSEED.
  6  Sampling a next token from the trained Tiny's shares for the running sentence: twenty draws with seed 1865, again with
     1865, then with 7.
Writes outputs/seeds_demo.json."""
import _common
import os, sys, json, subprocess
import numpy as np
import torch
torch.set_num_threads(1)
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from tiny.seed import fingerprint, streams
from tiny import data as D
from _pool import shards
from _common import HERE, SEED, sentence_ids, tok
O = {}
def fp_of(m):
    return fingerprint(TorchTiny.from_numpy(m, dtype=torch.float32).t)
a = fp_of(Tiny(seed=1865)); torch.manual_seed(12345); b = fp_of(Tiny(seed=1865)); c = fp_of(Tiny(seed=7))
O["1"] = dict(seed1865=a, seed1865_after_torch_seed=b, seed7=c)
print(f"1  Tiny(seed=1865) {a}; again after torch.manual_seed(12345) {b}; Tiny(seed=7) {c}")
first = {}
for s in (1865, 1865, 7):
    X = D.ShardStream(shards(), window=129, batch=8, buffer=256, seed=s).next_batch()
    first.setdefault(str(s), []).append([int(v) for v in X[:, 0]])
O["2"] = first
print("2  first token of each of the 8 windows in the first batch:", first)
def masks_shared(extra):
    g = np.random.default_rng(SEED)
    g.normal(size=1000)                       # the starting tables
    if extra:
        g.normal(size=1)                      # one extra draw at the start
    return (g.random((4, 16)) < 0.1).astype(int)
def masks_streams(extra):
    s = streams(SEED)
    s["init"].normal(size=1000)
    if extra:
        s["init"].normal(size=1)
    return (s["dropout"].random((4, 16)) < 0.1).astype(int)
m0, m1 = masks_shared(False), masks_shared(True)
n0, n1 = masks_streams(False), masks_streams(True)
O["3"] = dict(shared_changed=int((m0 != m1).sum()), streams_changed=int((n0 != n1).sum()), cells=int(m0.size),
              shared_dropped=[int(m0.sum()), int(m1.sum())], streams_dropped=[int(n0.sum()), int(n1.sum())])
print(f"3  dropout masks (64 cells, rate 0.1) after one extra draw: shared generator {O['3']['shared_changed']} cells changed; "
      f"one generator per job {O['3']['streams_changed']} changed")
def tmask(extra):
    torch.manual_seed(SEED)
    if extra:
        torch.rand(1)
    return torch.nn.functional.dropout(torch.ones(4, 16), 0.1).ne(0).int()
O["4"] = dict(same_twice=bool(torch.equal(tmask(False), tmask(False))), changed_after_extra=int((tmask(False) != tmask(True)).sum()))
print(f"4  torch dropout mask with manual_seed(1865) twice identical: {O['4']['same_twice']}; cells changed after one extra torch.rand: {O['4']['changed_after_extra']}")
code = "print(list({'the','cat','sat','on','mat','alice','queen','hatter'}))"
runs = {}
for label, env in (("no PYTHONHASHSEED", {}), ("PYTHONHASHSEED=0", {"PYTHONHASHSEED": "0"})):
    e = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}; e.update(env)
    runs[label] = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e).stdout.strip() for _ in range(3)]
O["5"] = {k: dict(orders=v, distinct=len(set(v))) for k, v in runs.items()}
for k, v in O["5"].items():
    print(f"5  set order in 3 fresh processes, {k}: {v['distinct']} distinct orders; first {v['orders'][0]}")
from tiny.eval import load
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
T = tok(); ids = sentence_ids(T)
with torch.no_grad():
    p = torch.softmax(tt.scores(torch.tensor(ids[:-1])[None])[0, -1].double(), -1).numpy()
p = p / p.sum()
draws = {}
for s in ("1865", "1865b", "7"):
    g = np.random.default_rng(int(s.rstrip("b")))
    draws[s] = [T.decode([int(i)]) for i in g.choice(len(p), size=20, p=p)]
top = np.argsort(-p)[:5]
O["6"] = dict(context=T.decode(ids[:-1]), top5=[(T.decode([int(i)]), round(float(p[i]), 4)) for i in top], draws=draws,
              same_1865=draws["1865"] == draws["1865b"], agree_7=int(sum(a == b for a, b in zip(draws["1865"], draws["7"]))))
print("6  context", repr(O["6"]["context"]), "top five", O["6"]["top5"])
for s, d in draws.items():
    print(f"   seed {s}: {d}")
print(f"   seed 1865 twice identical: {O['6']['same_1865']}; seed 7 agrees with 1865 at {O['6']['agree_7']} of 20 draws")
json.dump(O, open(os.path.join(HERE, "outputs", "seeds_demo.json"), "w"), indent=1)
