"""Lab 35: Tiny's edge, the rotary arithmetic, the blocked forward pass, and two breakages.  CPU, one thread, 32-bit, seed 1865.
Makes no API call; reads ../../.env only to confirm that the key string appears in no file under 04-code/ch35.
  python3 lab35.py"""
from _common import *
import torch, glob, math
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
from tiny.model import Tiny
from tiny.context import token_losses, sliding_losses, by_band, unturned_pairs, untrained_rows
from _pool import corpora
res = []
def check(ok, text): res.append(ok); print(f"[{'pass' if ok else 'FAIL'}] {text}", flush=True)
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
print("Part 1. Tiny's position table")
first = untrained_rows(tt.t["P"], Tiny().P.table)
print(f"  first position row that never moved from its seed-1865 start: {first}")
check(first == 128, "Tiny's trained window is 128 positions: rows 128 onwards are untouched")
print("Part 2. one window of 512 against a sliding window of 128")
g = [int(i) for i in corpora()["glass"]][:512]
full = token_losses(tt.scores, g); slide = sliding_losses(tt.scores, g, 128)
B = [(0, 127), (128, 510)]; f, s = by_band(full, B), by_band(slide, B)
print(f"  first window of the novel, positions 0-127: {f['0-127']:.4f} / {s['0-127']:.4f}; 128-510: {f['128-510']:.4f} (one pass) against {s['128-510']:.4f} (sliding)")
check(abs(f["0-127"] - s["0-127"]) < 1e-4 and f["128-510"] > s["128-510"], "inside 128 the two agree; past 128 one long pass is worse than the sliding window")
print("Part 3. breakage: score Tiny in windows of 512 instead of 129")
ids = [int(i) for i in corpora()["held_b"]]; tk = tok()
def bpc(n):
    tot, ch = 0.0, 0
    for a in range(0, len(ids) - 1, n - 1):
        w = ids[a:a + n]
        if len(w) < 2: break
        tot += float(token_losses(tt.scores, w).sum()); ch += len(tk.decode(w[1:]))
    return tot / math.log(2) / ch
b129, b512 = bpc(129), bpc(512)
print(f"  held-out Carroll: windows of 129 {b129:.4f} bits per character; windows of 512 {b512:.4f}")
check(abs(b129 - 1.9812) < 1e-3 and b512 > b129 + 0.1, "a window longer than the trained one makes Tiny's score worse, with no error raised")
print("Part 4. rotary arithmetic for SmolLM2 (head width 64)")
u = [unturned_pairs(64, 100000, L) for L in (2048, 8192, 16384)]; u16 = unturned_pairs(64, 1600000, 16384)
print(f"  pairs that never complete a turn at 2,048 / 8,192 / 16,384 tokens, base 100,000: {u}; base 1,600,000 at 16,384: {u16}")
check(u == [15, 12, 10] and u16 == 14, "raising the base leaves more pairs inside a single turn")
print("Part 5. the blocked forward pass")
line = open(os.path.join(HERE, "outputs", "check_longfwd.txt")).read().strip(); gap = float(line.split()[-1])
print("  " + line)
check(gap < 1e-4, "longfwd.py agrees with tiny/load.py within 1e-4 per token across a block boundary")
print("Part 6. the key")
key = None
for l in open(os.path.join(HERE, "..", "..", ".env")):
    if l.startswith("ANTHROPIC_API_KEY="): key = l.split("=", 1)[1].strip().strip('"')
found = [p for p in glob.glob(os.path.join(HERE, "**", "*"), recursive=True) if os.path.isfile(p) and key and key.encode() in open(p, "rb").read()]
check(bool(key) and not found, "the API key was read from .env and its string appears in no file under 04-code/ch35")
print(f"\n{sum(res)} of {len(res)} checkpoints pass")
if not all(res): raise SystemExit(1)
