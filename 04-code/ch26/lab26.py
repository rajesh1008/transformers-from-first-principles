"""Lab 26: predict three numbers, break reproducibility five ways, and run a checkpoint.  Reads the outputs of
seeds_demo.py, fp_order.py, run_seeds.sh and analyse.py; makes no API call and trains nothing.  One thread, 32-bit.
Run from 04-code/ch26 after run_all.sh.  Stops with an error if a checkpoint fails."""
import _common
import os, json, math, glob, subprocess, sys
import numpy as np
import torch
torch.set_num_threads(1)
from _common import HERE, SEED
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from tiny.seed import fingerprint, streams, ulp_distance, run_card
J = lambda n: json.load(open(os.path.join(HERE, "outputs", n + ".json")))
A, F, S = J("analyse"), J("fp_order"), J("seeds_demo")
print("Part 1. predict, then read")
f = np.float32
print(f"  (1e8 + 1) - 1e8 in 32-bit: {float((f(1e8) + f(1)) - f(1e8))}")
sd = A["five_glass"]["sd_both"]
print(f"  spread of the average of three seeds when one seed's spread is {sd:.4f}: {sd / math.sqrt(3):.4f}")
si, so = A["five_glass"]["sd_init"], A["five_glass"]["sd_order"]
print(f"  spreads {si:.4f} (start) and {so:.4f} (order) added as squares: {math.sqrt(si ** 2 + so ** 2):.4f}; measured with both changed {sd:.4f}")
print("\nPart 2. five breakages")
fps = set()
for s in (1, 2, 3, 4, 5):
    torch.manual_seed(s)
    fps.add(fingerprint(TorchTiny.from_numpy(Tiny(), dtype=torch.float32).t))
print(f"  1 'five seeds' by torch.manual_seed only: {len(fps)} distinct starting fingerprint(s)")
print(f"  2 one shared generator: {S['3']['shared_changed']} of 64 dropout cells moved by one unrelated draw (per-job streams: {S['3']['streams_changed']})")
d2 = A["drift"]["i1865_o1865_t4"]; d3 = A["drift"]["i1865_o1865_t4_r2"]
print(f"  3 four threads: batch losses part from one thread's at step {d2['first_step']}; two four-thread runs end with fingerprints "
      f"{d2['fp']} and {d3['fp']}")
L = A["lr_glass"]
print(f"  4 one seed per recipe: lr 1.4e-3 minus 1e-3 on Looking-Glass {', '.join(f'{v:+.4f}' for v in L['diffs'])}; "
      f"{L['unpaired_25_wrong']} of 25 single-run pairings point the other way from the five-seed mean {L['mean']:+.4f}")
print(f"  5 set order without PYTHONHASHSEED: {S['5']['no PYTHONHASHSEED']['distinct']} orders in 3 processes")
print("\nPart 3. checkpoints")
ok = []
def check(c, msg):
    ok.append(bool(c)); print(("[pass] " if c else "[FAIL] ") + msg)
r = A["repeats"]
check(len({v["fp"] for v in r.values()}) == 1, f"three one-thread runs of one configuration end with one fingerprint: {list(r.values())[0]['fp']}")
if "repeat_ch24" in A:
    check(A["repeat_ch24"]["identical"], f"Chapter 24's 2,500-step run, repeated from nothing, matches its saved tables byte for byte: {A['repeat_ch24']['fp_repeat']}")
a = F["A"]
check(abs(a["pairwise"] - a["exact"]) < abs(a["left_to_right"] - a["exact"]), f"pairwise summation beats left to right: error {a['pairwise'] - a['exact']:+.6f} against {a['left_to_right'] - a['exact']:+.6f}")
s0, s1 = streams(SEED), streams(SEED, jobs=("init", "order", "dropout", "sample", "extra"))
check(all(s0[k].integers(0, 2 ** 31) == s1[k].integers(0, 2 ** 31) for k in s0), "adding a job to streams() leaves the existing jobs' generators unchanged")
x = np.float32(1.0)
check(int(ulp_distance(x, np.nextafter(x, np.float32(2)))) == 1, "ulp_distance(1, next float32 above 1) = 1")
key = None
envp = os.path.join(HERE, "..", "..", ".env")
if os.path.exists(envp):
    for l in open(envp):
        if l.startswith("ANTHROPIC_API_KEY="):
            key = l.split("=", 1)[1].strip().strip('"')
found = []
if key:
    for p in glob.glob(os.path.join(HERE, "**", "*"), recursive=True):
        if os.path.isfile(p) and "/ckpt/" not in p:
            try:
                if key in open(p, errors="ignore").read():
                    found.append(p)
            except Exception:
                pass
check(key is not None and not found, "the API key string appears in no file under 04-code/ch26")
print(f"\n{sum(ok)} of {len(ok)} checkpoints pass")
if not all(ok):
    sys.exit(1)
