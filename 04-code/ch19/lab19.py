"""Lab 19: initialise Tiny three ways, break the start four ways, and check the activation spread by layer.
Part 1: the spread of the running sentence's rows entering each of Tiny's twelve layers (and leaving the last),
under the seeded default (0.02), the residual-scaled start and the fan-in start.  64-bit, seed 1865.
Part 2: four breakages.  Part 3: the checkpoints."""
import _common
import hashlib
import numpy as np
from tiny.model import Tiny
from tiny.init import initialise, spreads
from _common import sentence_ids, novel_ids, tok, SEED, probe
T = tok(); ids = sentence_ids(T)
def fingerprint(model):
    h = hashlib.sha256()
    for k, v in model.tables().items():
        h.update(k.encode()); h.update(np.ascontiguousarray(v, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]
print(f"Part 1. the running sentence {ids} (its first six seats, each predicting the next token): spread of the rows entering layers 0 to 11, and leaving layer 11")
runs = {}
for scheme in ("gpt2", "scaled", "fan_in"):
    P = probe(initialise(Tiny(), scheme), ids)
    runs[scheme] = P
    print(f"  {scheme:7s} " + " ".join(f"{v:.3f}" for v in P["stream"]) + f"   loss {P['loss']:.4f}")
print("\nPart 2. breakages")
P10 = probe(initialise(Tiny(), "gpt2", 10.0), ids)
print(f"  scale x10: stream leaving layer 11 {P10['stream'][-1]:.2f} (default {runs['gpt2']['stream'][-1]:.3f});"
      f" layer-0 attention entropy {P10['entropy'][0]:.3f} (default {runs['gpt2']['entropy'][0]:.3f})")
mz = initialise(Tiny(), "all_zeros")
from tiny.torch_tiny import TorchTiny
Lz, Gz = TorchTiny.from_numpy(mz).grads(ids)
print(f"  all zeros: loss {Lz:.4f}, largest nudge {max(np.abs(v).max() for v in Gz.values()):.1f}")
mc = initialise(Tiny(), "constant")
W1 = mc.blocks[0].ff.W_1
print(f"  constant 0.02: distinct neurons in layer 0's W_1 {len(np.unique(W1, axis=1).T)} of {W1.shape[1]}")
deep = {s: probe(initialise(Tiny(n_layers=48), s), ids)["stream"] for s in ("gpt2", "scaled")}
print(f"  48 layers, unscaled: stream grows {deep['gpt2'][-1] / deep['gpt2'][0]:.1f}x; scaled: {deep['scaled'][-1] / deep['scaled'][0]:.2f}x")
print("\ncheckpoints:")
checks = []
def check(name, ok):
    checks.append(ok); print(f"  [{'pass' if ok else 'FAIL'}] {name}")
m = initialise(Tiny(), "gpt2"); lt = m.loss(novel_ids(T)[:1025])
check(f"initialise(m, 'gpt2') changes nothing: {m.n_params:,} learned numbers, fingerprint {fingerprint(m)}, loss {lt:.4f}",
      m.n_params == 2766592 and fingerprint(m) == "19c8710677873d01" and round(lt, 4) == 7.6334)
g = runs["gpt2"]["stream"]
check(f"seeded start, spread by layer: {g[0]:.4f} entering layer 0, {g[1]:.4f} at layer 1, {g[6]:.4f} at layer 6, {g[12]:.4f} out",
      [round(g[k], 4) for k in (0, 1, 6, 12)] == [0.0227, 0.0673, 0.1771, 0.2445])
sp = spreads(initialise(Tiny(), "fan_in"))
check(f"fan-in start: W_q {sp['att.W_q']:.4f} against 1/sqrt(128) {1/np.sqrt(128):.4f}, W_2 {sp['ff.W_2']:.4f} against 1/sqrt(512) {1/np.sqrt(512):.4f}",
      abs(sp["att.W_q"] * np.sqrt(128) - 1) < 0.03 and abs(sp["ff.W_2"] * np.sqrt(512) - 1) < 0.03)
check(f"scale x10 collapses layer-0 attention: entropy {P10['entropy'][0]:.3f} below half the default's",
      P10["entropy"][0] < 0.5 * runs["gpt2"]["entropy"][0])
check("all zeros: every nudge is exactly zero", all((v == 0).all() for v in Gz.values()))
check("constant start: one distinct neuron", len(np.unique(W1, axis=1).T) == 1)
check(f"48 layers: scaled growth {deep['scaled'][-1] / deep['scaled'][0]:.2f}x under 3x, unscaled {deep['gpt2'][-1] / deep['gpt2'][0]:.1f}x over 10x",
      deep["scaled"][-1] / deep["scaled"][0] < 3 and deep["gpt2"][-1] / deep["gpt2"][0] > 10)
print(f"\n{sum(checks)} of {len(checks)} checkpoints pass")
assert all(checks)
