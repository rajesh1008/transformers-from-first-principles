"""Chapter 17, section 17.6: training the hand Tiny on the running sentence with tiny/train.py, twice:
once with nudges from wiggling (NumPy only) and once with autograd (PyTorch), same start, same learning
rate.  The two runs should agree to the wiggle's precision at every step.  Then: what the trained hand
Tiny predicts.  64-bit; nothing random."""
import _common
import time
import numpy as np
from hand import hand_tiny, IDS, WORDS
from tiny.train import train
from _common import save_json, r3

STEPS, LR = 200, 0.5
runs = {}
for how in ("wiggle", "torch"):
    m = hand_tiny()
    t0 = time.perf_counter()
    hist = train(m, lambda s: IDS, STEPS, LR, grad_fn=how)
    dt = time.perf_counter() - t0
    runs[how] = dict(loss=[L for _, L in hist], final=m.loss(IDS), seconds=dt, model=m)
    print(f"{how:>6}: loss {hist[0][1]:.4f} -> step 10 {hist[10][1]:.4f} -> step 50 {hist[50][1]:.4f} -> step 199 {hist[-1][1]:.4f};"
          f" after the last nudge {m.loss(IDS):.4f}; {dt:.1f} s")
d = max(abs(a - b) for a, b in zip(runs["wiggle"]["loss"], runs["torch"]["loss"]))
T_ = runs["torch"]["loss"]
peaks = [(i, T_[i]) for i in range(1, len(T_) - 1) if T_[i] > T_[i - 1] and T_[i] > T_[i + 1] and T_[i] > 1.5 * min(T_[max(0, i - 5):i])]
print("spikes in the lr 0.5 run (a peak 1.5 times the lowest of the five steps before it):",
      "; ".join(f"step {i} loss {v:.3f} (from {min(T_[max(0, i - 5):i]):.3f})" for i, v in peaks))
print(f"largest difference between the two loss curves over {STEPS} steps: {d:.2e}")
m = runs["torch"]["model"]
S = m.forward(IDS)
P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
print("\nthe trained hand Tiny on the running sentence:")
for s in range(5):
    top = np.argsort(-P[s])[:3]
    print(f"  after {WORDS[IDS[s]]}@{s}: top three " + ", ".join(f"{WORDS[i]} {P[s,i]:.3f}" for i in top) +
          f" | true next '{WORDS[IDS[s+1]]}' share {P[s, IDS[s+1]]:.3f}")
top = np.argsort(-P[5])[:3]
print("  after mat@5 (no true next word): " + ", ".join(f"{WORDS[i]} {P[5,i]:.3f}" for i in top))
h0 = hand_tiny().tables(); h1 = m.tables()
moved = sorted(((float(np.abs(h1[k] - h0[k]).max()), k) for k in h0), reverse=True)[:5]
print("tables that moved most (largest single change):", "; ".join(f"{k} {v:.3f}" for v, k in moved))
print("cat's row", r3(h0['E'][1]), "->", r3(h1['E'][1]), "| mat's row", r3(h0['E'][4]), "->", r3(h1['E'][4]))
print("\nlearning rates, 200 steps each (autograd):")
lrs = {}
for lr in (0.1, 0.5, 2.0, 8.0):
    mm = hand_tiny(); hist = train(mm, lambda s: IDS, STEPS, lr)
    Ls = [L for _, L in hist]; lrs[str(lr)] = Ls
    print(f"  lr {lr}: loss at steps 0, 10, 50, 199: {Ls[0]:.4g} {Ls[10]:.4g} {Ls[50]:.4g} {Ls[-1]:.4g}; largest {np.nanmax(Ls):.4g}; first step above 10: {next((k for k, v in enumerate(Ls) if not v < 10), None)}")
save_json("train_hand", dict(wiggle=runs["wiggle"]["loss"], torch=runs["torch"]["loss"], diff=d, lrs=lrs,
                             seconds={k: v["seconds"] for k, v in runs.items()}))
