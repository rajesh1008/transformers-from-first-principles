"""Chapter 26, sections 26.4 and 26.5: floating-point addition is not associative, and what that does to Tiny.
  Part A  one million 32-bit numbers (normal, seed 1865) added in different orders: left to right, NumPy's pairwise sum,
          smallest first, largest first, and 200 random orders; against the exactly rounded sum (math.fsum in 64-bit).
  Part B  the three-number case: (0.1 + 0.2) + 0.3 against 0.1 + (0.2 + 0.3), in 64-bit and 32-bit.
  Part C  batch invariance on this CPU: Chapter 24's trained Tiny (2,500 steps, 32-bit) scores the running sentence alone
          and as the first row of a batch of 8 and of 64 windows; largest difference in the scores and in the loss, and
          whether the top next-token choice changes.  Also a single 1 x 128 by 128 x 2048 product inside batches of 1 to 512 rows.
  Part D  thread counts: the loss and the gradient of the same batch of 8 windows at 1, 2, 4 and 8 threads, compared bit for
          bit with one thread (each count in a fresh process, so no state carries over).
Writes outputs/fp_order.json and prints the tables used in the chapter."""
import _common
import os, sys, json, math, subprocess
import numpy as np
from _common import HERE, SEED, sentence_ids
OUT = {}
if len(sys.argv) > 1 and sys.argv[1] == "threads":          # Part D worker: one thread count per process
    import torch
    torch.set_num_threads(int(sys.argv[2]))
    from tiny.eval import load
    from tiny import data as D
    from _pool import shards
    tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
    X = torch.tensor(D.ShardStream(shards(), window=129, batch=8, buffer=256, seed=SEED).next_batch())
    for v in tt.t.values():
        v.requires_grad_(True); v.grad = None
    L = tt.loss(X); L.backward()
    g = torch.cat([tt.t[k].grad.flatten() for k in sorted(tt.t)])
    np.save(os.path.join(HERE, "outputs", f"_grad_t{sys.argv[2]}.npy"), g.numpy())
    print(json.dumps(dict(threads=torch.get_num_threads(), loss=float(L), loss_hex=float(L).hex())))
    sys.exit()

# Part A
rng = np.random.default_rng(SEED)
x = rng.normal(0, 1, 1_000_000).astype(np.float32)
exact = math.fsum(x.astype(np.float64))
def seq(v):
    return float(np.cumsum(v, dtype=np.float32)[-1])        # strictly left to right, one rounding per addition
A = dict(exact=exact, left_to_right=seq(x), pairwise=float(np.sum(x, dtype=np.float32)),
         smallest_first=seq(x[np.argsort(np.abs(x))]), largest_first=seq(x[np.argsort(-np.abs(x))]))
orders = [seq(x[rng.permutation(x.size)]) for _ in range(200)]
A.update(random_orders=dict(n=200, distinct=len(set(orders)), min=min(orders), max=max(orders),
                             sd=float(np.std(orders)), mean=float(np.mean(orders))))
OUT["A"] = A
print("Part A  one million float32 numbers, sum in different orders")
for k in ("exact", "left_to_right", "pairwise", "smallest_first", "largest_first"):
    print(f"  {k:15s} {A[k]:.6f}   error {A[k] - exact:+.6f}")
r = A["random_orders"]
print(f"  200 random orders: {r['distinct']} distinct sums, from {r['min']:.6f} to {r['max']:.6f} (sd {r['sd']:.6f})")
# Part B
B = dict(f64=[(0.1 + 0.2) + 0.3, 0.1 + (0.2 + 0.3)])
f = np.float32
B["f32"] = [float((f(0.1) + f(0.2)) + f(0.3)), float(f(0.1) + (f(0.2) + f(0.3)))]
B["big_small"] = [float((f(1e8) + f(1)) - f(1e8)), float((f(1e8) - f(1e8)) + f(1))]
OUT["B"] = B
print("Part B  (0.1+0.2)+0.3 vs 0.1+(0.2+0.3):  64-bit", repr(B["f64"][0]), repr(B["f64"][1]),
      "| 32-bit", repr(B["f32"][0]), repr(B["f32"][1]), "| (1e8+1)-1e8 vs (1e8-1e8)+1 in 32-bit:", B["big_small"])
# Part C
import torch
torch.set_num_threads(1)
from tiny.eval import load
from tiny import data as D
from _pool import shards
from tiny.seed import ulp_distance
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
sent = torch.tensor(sentence_ids())
st = D.ShardStream(shards(), window=129, batch=8, buffer=256, seed=SEED)
others = torch.tensor(np.concatenate([st.next_batch() for _ in range(8)]))[:, :len(sent)]
C = {}
with torch.no_grad():
    s1 = tt.scores(sent[None, :-1])[0]
    for bsz in (8, 64):
        Xb = torch.cat([sent[None], others[:bsz - 1]])
        sb = tt.scores(Xb[:, :-1])[0]
        l1, lb = float(tt.loss(sent[None])), float(tt.loss(Xb[:1]))
        C[bsz] = dict(max_abs_diff=float((s1 - sb).abs().max()), n_diff=int((s1 != sb).sum()), n=int(s1.numel()),
                      argmax_same=bool((s1.argmax(-1) == sb.argmax(-1)).all()),
                      max_ulp=int(ulp_distance(s1.numpy(), sb.numpy()).max()))
    W = tt.t["E"].T.contiguous()
    rows = torch.tensor(np.random.default_rng(SEED).normal(0, 1, (512, 128)).astype(np.float32))
    one = rows[:1] @ W
    C["matmul"] = {int(b): float(((rows[:b] @ W)[:1] - one).abs().max()) for b in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512)}
OUT["C"] = C
print("Part C  the sentence's scores alone vs first row of a batch (2,500-step Tiny, one thread)")
for bsz in (8, 64):
    c = C[bsz]
    print(f"  batch {bsz:3d}: {c['n_diff']} of {c['n']} scores differ, largest {c['max_abs_diff']:.3g} ({c['max_ulp']} ulp); top choice same at every seat: {c['argmax_same']}")
mm = C["matmul"]
print(f"  one row times E^T alone and inside batches of b rows: b = 1 difference {mm[1]:g}; b = 2 to 512 largest difference "
      f"{max(v for b, v in mm.items() if b > 1):.3g}, smallest {min(v for b, v in mm.items() if b > 1):.3g}")
# Part D
D_ = []
for th in (1, 2, 4, 8):
    r = subprocess.run([sys.executable, __file__, "threads", str(th)], capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS=str(th), MKL_NUM_THREADS=str(th)))
    D_.append(json.loads(r.stdout.strip().splitlines()[-1]))
g1 = np.load(os.path.join(HERE, "outputs", "_grad_t1.npy"))
for d in D_:
    gt = np.load(os.path.join(HERE, "outputs", f"_grad_t{d['threads']}.npy"))
    d.update(loss_diff=d["loss"] - D_[0]["loss"], grad_n_diff=int((gt != g1).sum()), grad_n=int(g1.size),
             grad_max_diff=float(np.abs(gt - g1).max()), grad_rel=float(np.linalg.norm(gt - g1) / np.linalg.norm(g1)))
for th in (1, 2, 4, 8):
    os.remove(os.path.join(HERE, "outputs", f"_grad_t{th}.npy"))
OUT["D"] = D_
print("Part D  one batch's loss and gradient at 1, 2, 4, 8 threads (fresh process each)")
for d in D_:
    print(f"  {d['threads']} threads: loss {d['loss']:.9f} (diff {d['loss_diff']:+.3g}); gradient numbers differing from 1 thread "
          f"{d['grad_n_diff']:,} of {d['grad_n']:,}, largest {d['grad_max_diff']:.3g}, relative size {d['grad_rel']:.3g}")
OUT["torch"] = torch.__version__
json.dump(OUT, open(os.path.join(HERE, "outputs", "fp_order.json"), "w"), indent=1)
