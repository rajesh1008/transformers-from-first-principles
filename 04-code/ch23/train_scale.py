"""Chapter 23: one of the five Tinys (sizes.py) trained for STEPS steps of 8 windows of 129 tokens (1,024 predictions a
step) read from the Carroll pool through ShardStream (24 shards of shuffled 4,096-token pieces, shuffle buffer 256,
order seed ORDER_SEED).  Mixed BF16 with a
32-bit master copy (Chapter 21), Adam (b2 0.95) at LR, warmup min(40, STEPS/10) then cosine to 0.1 of LR over the run.
Held-out loss in 32-bit on Chapter 22's two sets (Carroll held-out, Looking-Glass), at nine points along the run,\nand at the end the loss on the running sentence.
Starts: Chapter 14's, seed 1865, for every run (only the reading order changes with ORDER_SEED).  One thread.
Usage: python3 train_scale.py SIZE STEPS [ORDER_SEED] [LR]     writes outputs/train_SIZE_STEPS[_sSEED][_lrLR].json
D (tokens read) = STEPS x 1,024;  compute = 6 N D with N the non-embedding count (the convention of the 2020 paper)."""
import _common
import sys, time, datetime, json, os
import numpy as np
import torch
torch.set_num_threads(1)
torch.manual_seed(1865)
from tiny.torch_tiny import TorchTiny
from tiny.optim import Adam, warmup_cosine
from tiny.precision import MixedTiny, train_mixed
from tiny import data as D
from _common import SEED, HERE
from _pool import heldout_sets, shards
from sizes import make, counts, TOKENS_PER_STEP
SIZE, STEPS = sys.argv[1], int(sys.argv[2])
OSEED = int(sys.argv[3]) if len(sys.argv) > 3 else SEED
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
TAG = f"{SIZE}_{STEPS}" + (f"_s{OSEED}" if OSEED != SEED else "") + (f"_lr{sys.argv[4]}" if len(sys.argv) > 4 else "")
TAG += os.environ.get("TAG_SUFFIX", "")                  # for repeated runs of one configuration (repro.sh)
WU = min(40, max(1, STEPS // 10))
G, HB = heldout_sets()
G, HB = torch.tensor(G), torch.tensor(HB)
stream = D.ShardStream(shards(), window=129, batch=8, buffer=256, seed=OSEED)
X = torch.tensor(np.stack([stream.next_batch() for _ in range(STEPS)]))
passes = stream.served * 8 / 3055                  # 3,055 windows in one pass of the 24 shards (prep.py)
m = make(SIZE)
tt = MixedTiny.from_numpy(m, compute=torch.bfloat16)
opt = Adam(LR)
def held(H):
    ref = TorchTiny({k: v.detach().to(torch.float32).numpy() for k, v in tt.t.items()}, tt.config, dtype=torch.float32)
    with torch.no_grad():
        return float(ref.loss(H))
marks = sorted(set([int(round(STEPS * k / 8)) - 1 for k in range(1, 9)]))
curve = [(-1, None, held(HB), held(G))]
def on_step(s, L):
    if s in marks:
        curve.append((s, L, held(HB), held(G)))
t0 = time.perf_counter()
hist = train_mixed(tt, lambda s: X[s], STEPS, opt, warmup_cosine(WU, STEPS, 0.1), mode="mixed", on_step=on_step)
dt = time.perf_counter() - t0
from _common import sentence_ids
sent = torch.tensor(sentence_ids())
sent_loss = held(sent)                     # the running sentence, "the cat sat on the mat", average surprise of its next tokens
c = counts(SIZE)
Dtok = STEPS * TOKENS_PER_STEP
out = dict(tag=TAG, size=SIZE, width=c["width"], N=c["non_embedding"], N_total=c["total"], steps=STEPS, D=Dtok,
           C=6 * c["non_embedding"] * Dtok, C_total=6 * c["total"] * Dtok, passes=passes, order_seed=OSEED, lr=LR,
           warmup=WU, date=str(datetime.date.today()), threads=torch.get_num_threads(),
           carroll=curve[-1][2], glass=curve[-1][3], sentence=sent_loss, sentence_ids=sent.tolist(), curve=curve, train=[L for _, L in hist], seconds=dt,
           ms_per_step=1000 * dt / STEPS)
json.dump(out, open(os.path.join(HERE, "outputs", f"train_{TAG}.json"), "w"), indent=1)
print(f"{TAG:18s} N {c['non_embedding']:>8,d} D {Dtok:>9,d} C {out['C']:.3e} passes {passes:.2f}  held-out Carroll "
      f"{out['carroll']:.4f}  Looking-Glass {out['glass']:.4f}  sentence {sent_loss:.4f}  last-24 batch {np.mean(out['train'][-24:]):.4f}  "
      f"{dt:.0f} s ({out['ms_per_step']:.0f} ms/step)")
