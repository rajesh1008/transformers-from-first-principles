"""Chapter 34, 34.7: does removing feed-forward units make Tiny faster?  The units removed in 34.5 are cut out of the tables
(W_1 loses columns, b_1 entries, W_2 rows), so the matrices really are smaller, and the forward pass over 8 windows of 128 tokens
is timed (median of 15 after 3 warm-up passes) against the full model and against the zeroed-but-full-size model.  The sliced and
zeroed models must give the same scores.  CPU, one thread, 32-bit, seed 1865."""
from _common import *
import torch, time, statistics, math
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
from tiny.compress import unit_masks, apply_masks
CK = os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")
X = torch.randint(0, 2048, (8, 128), generator=torch.Generator().manual_seed(SEED))
def timed(m):
    with torch.no_grad():
        for _ in range(3): m.scores(X)
        ts = []
        for _ in range(15):
            t0 = time.perf_counter(); m.scores(X); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000
full = load(CK); R = {"full ms": timed(full)}
print(f"full model, 512 feed-forward units per layer: {R['full ms']:.1f} ms per forward pass (8 x 128 tokens)")
for frac in (0.5, 0.75):
    z = load(CK); masks = unit_masks(z.t, 12, frac); apply_masks(z.t, masks)
    s = load(CK)
    for l in range(12):
        p = f"h.{l}.ff."; keep = masks[p + "b_1"].nonzero().flatten()
        s.t[p + "W_1"] = s.t[p + "W_1"][:, keep].contiguous(); s.t[p + "b_1"] = s.t[p + "b_1"][keep].contiguous()
        s.t[p + "W_2"] = s.t[p + "W_2"][keep, :].contiguous()
    with torch.no_grad(): gap = float((z.scores(X) - s.scores(X)).abs().max())
    tz, ts = timed(z), timed(s); n = sum(v.numel() for v in s.t.values())
    R[f"{frac} zeroed ms"], R[f"{frac} sliced ms"], R[f"{frac} gap"], R[f"{frac} numbers"] = tz, ts, gap, n
    print(f"{int(frac * 100)}% of units removed: zeroed full-size {tz:.1f} ms; sliced ({s.t['h.0.ff.W_1'].shape[1]} units, {n:,} numbers) "
          f"{ts:.1f} ms ({R['full ms'] / ts:.2f} x the full model); largest score gap zeroed vs sliced {gap:.1e}")
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
four = TorchTiny.from_numpy(Tiny(n_layers=4), dtype=torch.float32); R["4-layer ms"] = timed(four)
print(f"the four-layer student of 34.2 (1,184,512 numbers): {R['4-layer ms']:.1f} ms ({R['full ms'] / R['4-layer ms']:.2f} x the full model)")
save_json("time_sliced", R)
