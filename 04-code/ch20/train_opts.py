"""Chapter 20: Tiny (2,766,592 learned numbers) trained for 400 steps by different optimisers, schedules and starts.
Same windows as Chapter 19's train_inits.py: eight windows of 129 tokens of the novel per step, drawn once with
seed 1865; held-out loss on the same 16 windows of the Looking-Glass text every 25 steps.  32-bit.
Usage: python3 train_opts.py GROUP THREADS   (GROUP is opt, sched, starts or tune; each writes outputs/train_GROUP.json)"""
import _common
import sys, time, datetime
import numpy as np
import torch
from tiny.model import Tiny
from tiny.init import initialise
from tiny.torch_tiny import TorchTiny
from tiny.optim import SGD, Adam, Muon, train_opt, constant, warmup_cosine, wsd, linear_decay
from _common import novel_ids, heldout_ids, tok, save_json, SEED
GROUP, TH = sys.argv[1], int(sys.argv[2])
torch.set_num_threads(TH)
T = tok()
nov = np.array(novel_ids(T)); held = np.array(heldout_ids(T))
STEPS, EVERY, WU = 400, 25, 40
rng = np.random.default_rng(SEED)
starts = rng.integers(0, len(nov) - 129, size=(STEPS, 8))
hstarts = rng.integers(0, len(held) - 129, size=16)
H = torch.tensor(np.stack([held[a:a + 129] for a in hstarts]))
def batches(s):
    return torch.tensor(np.stack([nov[a:a + 129] for a in starts[s]]))
wc = lambda: warmup_cosine(WU, STEPS, 0.1)
RUNS = {
 "opt": [
  ("sgd 0.1", "gpt2", 1.0, lambda: SGD(0.1), constant),
  ("sgd 0.6", "gpt2", 1.0, lambda: SGD(0.6), constant),
  ("momentum 0.9, lr 0.01", "gpt2", 1.0, lambda: SGD(0.01, 0.9), constant),
  ("momentum 0.9, lr 0.06", "gpt2", 1.0, lambda: SGD(0.06, 0.9), constant),
  ("adam 1e-3 constant", "gpt2", 1.0, lambda: Adam(1e-3), constant),
  ("adam 3e-3 constant", "gpt2", 1.0, lambda: Adam(3e-3), constant),
  ("adam 1e-2 constant", "gpt2", 1.0, lambda: Adam(1e-2), constant),
 ],
 "sched": [
  ("adam 3e-3 warmup+cosine", "gpt2", 1.0, lambda: Adam(3e-3), wc),
  ("adam 1e-2 warmup+cosine", "gpt2", 1.0, lambda: Adam(1e-2), wc),
  ("adam 3e-2 warmup+cosine", "gpt2", 1.0, lambda: Adam(3e-2), wc),
  ("adam 1e-3 warmup+cosine", "gpt2", 1.0, lambda: Adam(1e-3), wc),
  ("adam 3e-3 wsd", "gpt2", 1.0, lambda: Adam(3e-3), lambda: wsd(WU, STEPS, 0.2)),
  ("adam 3e-3 linear to 0", "gpt2", 1.0, lambda: Adam(3e-3), lambda: linear_decay(WU, STEPS)),
  ("adamw 3e-3 wd 0.1 warmup+cosine", "gpt2", 1.0, lambda: Adam(3e-3, wd=0.1), wc),
 ],
 "starts": [
  ("muon 3e-3 warmup+cosine", "gpt2", 1.0, lambda: Muon(3e-3, adam=Adam(3e-3)), wc),
  ("adam 3e-3 warmup+cosine, scaled start", "scaled", 1.0, lambda: Adam(3e-3), wc),
  ("adam 3e-3 warmup+cosine, fan-in start", "fan_in", 1.0, lambda: Adam(3e-3), wc),
  ("adam 3e-3 warmup+cosine, x10 start", "gpt2", 10.0, lambda: Adam(3e-3), wc),
  ("adam 3e-3 warmup+cosine, layer tables zero", "zeros", 1.0, lambda: Adam(3e-3), wc),
  ("adam 1e-2 constant, clip 1", "gpt2", 1.0, lambda: Adam(1e-2), constant),
 ],
 "tune": [
  ("muon 1e-3 warmup+cosine", "gpt2", 1.0, lambda: Muon(1e-3, adam=Adam(1e-3)), wc),
  ("muon 1e-2 warmup+cosine", "gpt2", 1.0, lambda: Muon(1e-2, adam=Adam(1e-2)), wc),
 ],
}
print(f"{datetime.date.today()}; group {GROUP}; seed {SEED}; threads {torch.get_num_threads()}; {STEPS} steps x 8 windows x 128 predictions; warmup {WU}")
out = {}
for desc, scheme, scale, mk, sch in RUNS[GROUP]:
    m = initialise(Tiny(), scheme, scale)
    tt = TorchTiny.from_numpy(m, dtype=torch.float32)
    curve, norms = [], []
    def on_step(s, L):
        if s % EVERY == 0 or s == STEPS - 1:
            with torch.no_grad():
                curve.append((s, L, float(tt.loss(H))))
    with torch.no_grad():
        h0 = float(tt.loss(H))
    clip = 1.0 if "clip 1" in desc else None
    t0 = time.perf_counter()
    hist = train_opt(tt, batches, STEPS, mk(), sch(), clip=clip, norms=norms, on_step=on_step)
    dt = time.perf_counter() - t0
    ok = len(hist) == STEPS and np.isfinite(hist[-1][1])
    with torch.no_grad():
        hend = float(tt.loss(H)) if ok else float("nan")
    best = min(c[2] for c in curve) if curve else float("nan")
    out[desc] = dict(scheme=scheme, scale=scale, curve=curve, train=[L for _, L in hist], norms=norms,
                     stopped=len(hist), held0=h0, held_end=hend, best=best, seconds=dt)
    last = np.mean([L for _, L in hist[-25:]])
    if not ok:
        print(f"{desc:44s} stopped at step {len(hist) - 1}: loss not finite; best held-out before {best:.4f}")
    else:
        print(f"{desc:44s} held-out {h0:.4f} -> {hend:.4f} (lowest {best:.4f}); batch last 25 {last:.4f};"
              f" norm step 0 {norms[0]:.3g}, largest {max(norms):.3g}; {dt:.0f} s")
    sys.stdout.flush()
save_json("train_" + GROUP, out)
