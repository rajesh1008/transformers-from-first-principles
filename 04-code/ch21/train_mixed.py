"""Chapter 21: Tiny (2,766,592 learned numbers) trained for 400 steps in different number formats.
Same windows as Chapters 19 and 20 (eight windows of 129 tokens per step, drawn once with seed 1865; held-out
loss on the same 16 windows of the Looking-Glass text every 25 steps).  Adam lr 1e-3, warmup 40 and cosine to 0.1.
Held-out loss is always computed in 32-bit from the tables the run keeps (for BF16 tables, their exact values).
One thread per run (so a run can be repeated bit for bit); runs go side by side as separate processes.
Usage: python3 train_mixed.py RUN [LR const]   (RUN as in RUNS below; writes outputs/train_mixed_RUN.json;
       with LR and "const", a constant learning rate and outputs/train_mixed_RUN_lrLR.json)"""
import _common
import sys, time, datetime, json, os
import numpy as np
import torch
torch.set_num_threads(1)
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from tiny.optim import Adam, warmup_cosine
from tiny.precision import MixedTiny, LossScaler, Adam16, train_mixed
from _common import novel_ids, heldout_ids, tok, SEED, HERE
RUN = sys.argv[1]
T = tok()
nov = np.array(novel_ids(T)); held = np.array(heldout_ids(T))
STEPS, EVERY, WU, LR = 400, 25, 40, 1e-3
CONST = len(sys.argv) > 3 and sys.argv[3] == "const"
if len(sys.argv) > 2:
    LR = float(sys.argv[2])
TAG = RUN + (f"_lr{sys.argv[2]}" if len(sys.argv) > 2 else "")
rng = np.random.default_rng(SEED)
starts = rng.integers(0, len(nov) - 129, size=(STEPS, 8))
hstarts = rng.integers(0, len(held) - 129, size=16)
H = torch.tensor(np.stack([held[a:a + 129] for a in hstarts]))
def batches(s):
    return torch.tensor(np.stack([nov[a:a + 129] for a in starts[s]]))
RUNS = {
 "fp32":        "32-bit throughout (Chapter 20's arithmetic)",
 "bf16":        "mixed: BF16 products, 32-bit master tables, Adam and loss",
 "fp16":        "mixed: FP16 products, no loss scaling",
 "fp16_scaled": "mixed: FP16 products, dynamic loss scaling from 2^16",
 "bf16_state":  "mixed BF16, with Adam's m and v stored in BF16",
 "w16_nearest": "BF16 tables, no master copy, rounded to nearest",
 "w16_sr":      "BF16 tables, no master copy, stochastic rounding",
}
m = Tiny()
if RUN.startswith("w16"):
    tt = TorchTiny.from_numpy(m, dtype=torch.bfloat16)
elif RUN == "fp32":
    tt = TorchTiny.from_numpy(m, dtype=torch.float32)
else:
    tt = MixedTiny.from_numpy(m, compute=torch.float16 if RUN.startswith("fp16") else torch.bfloat16)
opt = Adam16(LR) if RUN == "bf16_state" else Adam(LR)
scaler = LossScaler(2.0 ** 16, interval=100) if RUN == "fp16_scaled" else None
gen = torch.Generator().manual_seed(SEED)
def held_loss():
    ref = TorchTiny({k: v.detach().to(torch.float32).numpy() for k, v in tt.t.items()}, tt.config, dtype=torch.float32)
    with torch.no_grad():
        return float(ref.loss(H))
curve, stats = [], []
def on_step(s, L):
    if s % EVERY == 0 or s == STEPS - 1:
        curve.append((s, L, held_loss()))
h0 = held_loss()
t0 = time.perf_counter()
from tiny.optim import constant
hist = train_mixed(tt, batches, STEPS, opt, constant() if CONST else warmup_cosine(WU, STEPS, 0.1),
                   mode="weights16" if RUN.startswith("w16") else "mixed", scaler=scaler,
                   stochastic=(RUN == "w16_sr"), gen=gen, on_step=on_step, stats=stats)
dt = time.perf_counter() - t0
ok = len(hist) == STEPS and np.isfinite(hist[-1][1])
hend = held_loss() if ok else float("nan")
zero_frac = None
out = dict(run=RUN, lr=LR, schedule="constant" if CONST else "warmup 40 + cosine to 0.1", desc=RUNS[RUN], date=str(datetime.date.today()), seed=SEED, threads=torch.get_num_threads(),
           held0=h0, held_end=hend, best=min(c[2] for c in curve) if curve else None, curve=curve,
           train=[L for _, L in hist], stopped=len(hist), seconds=dt,
           lost=stats if stats else None,
           scaler=dict(skipped=scaler.skipped, final=scaler.scale, history=scaler.history) if scaler else None)
json.dump(out, open(os.path.join(HERE, "outputs", f"train_mixed_{TAG}.json"), "w"), indent=1)
last = np.mean(out["train"][-25:])
extra = ""
if stats:
    extra = f"; lost updates step 1 {stats[0]:.3f}, step 200 {stats[min(199, len(stats)-1)]:.3f}, last {stats[-1]:.3f}"
if scaler:
    extra = f"; skipped steps {scaler.skipped}, final scale 2^{int(np.log2(scaler.scale))}"
print(f"{TAG:12s} {RUNS[RUN]:58s} lr {LR:g}{' constant' if CONST else ''}; held-out {h0:.4f} -> {hend:.4f} (lowest {out['best']:.4f}); batch last 25 {last:.4f}; {dt:.0f} s{extra}")
