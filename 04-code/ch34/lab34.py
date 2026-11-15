"""Lab 34: check the distillation loss, the pruning masks, two breakages, and the recorded runs.  CPU, one thread, 32-bit,
seed 1865.  Makes no API call; reads ../../.env only to confirm that the key string appears in no file under 04-code/ch34.
  python3 lab34.py"""
from _common import *
import torch, math, glob
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
from tiny.optim import Adam
from tiny import data as D
from tiny.tokenizer import load_bpe
from _pool import corpora, shards
CK = os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")
tk = load_bpe(TINY_BPE); ids = [int(i) for i in corpora()["held_b"]]
def bpc(m):
    tot, chars = 0.0, 0
    for a in range(0, len(ids) - 1, 129):
        w = ids[a:a + 129]
        if len(w) < 2: break
        with torch.no_grad(): lp = torch.log_softmax(m.scores(w[:-1]).double(), -1)
        tot += float(-lp[torch.arange(len(w) - 1), torch.tensor(w[1:])].sum()); chars += len(tk.decode(w[1:]))
    return tot / math.log(2) / chars
res = []
def check(ok, text): res.append(ok); print(f"[{'pass' if ok else 'FAIL'}] {text}", flush=True)

print("Part 1. the distillation loss")
g = torch.Generator().manual_seed(SEED)
zs = torch.randn(5, 2048, generator=g, dtype=torch.float64, requires_grad=True); zt = torch.randn(5, 2048, generator=g, dtype=torch.float64)
lt = torch.log_softmax(zt, -1); L = (lt.exp() * (lt - torch.log_softmax(zs, -1))).sum(-1).sum(); L.backward()
diff = float((zs.grad - (torch.softmax(zs, -1) - torch.softmax(zt, -1))).abs().max())
self_kl = float((lt.exp() * (lt - lt)).sum())
print(f"  KL of a table with itself {self_kl:.1e}; largest gap between autograd and (student - teacher) {diff:.2e}")
check(diff < 1e-12 and self_kl == 0.0, "the KL gradient on the student's scores is the student's table minus the teacher's")

print("Part 2. masks")
tt = load(CK); base = bpc(tt); orig = {k: v.clone() for k, v in tt.t.items()}
mats = [k for k, v in orig.items() if v.dim() == 2 and k not in ("E", "P")]
counts = set()
for k in mats:
    w = orig[k]; idx = w.abs().argsort(0)[:w.shape[0] // 2]; m = torch.ones_like(w); m.scatter_(0, idx, 0.0); tt.t[k] = w * m
    counts |= set((tt.t[k] != 0).sum(0).tolist())
b50 = bpc(tt)
print(f"  32-bit {base:.4f}; half of every output channel's weights zeroed {b50:.4f} ({b50 - base:+.4f}); nonzero counts per channel {sorted(counts)}")
check(len(counts) <= 2 and b50 - base < 0.03, "per-channel pruning leaves every output channel the same count and costs Tiny under 0.03 at 50%")

print("Part 3. breakage: a unit removed on its input side only")
def units(frac, how):
    for k in orig: tt.t[k] = orig[k].clone()
    for l in range(tt.config["n_layers"]):
        p = f"h.{l}.ff."; W1, W2 = orig[p + "W_1"], orig[p + "W_2"]
        drop = (W1.norm(dim=0) * W2.norm(dim=1)).argsort()[:int(W1.shape[1] * frac)]
        a, b, c = W1.clone(), orig[p + "b_1"].clone(), W2.clone()
        if how in ("all", "in"): a[:, drop] = 0
        if how == "all": b[drop] = 0
        if how in ("all", "out"): c[drop, :] = 0
        tt.t[p + "W_1"], tt.t[p + "b_1"], tt.t[p + "W_2"] = a, b, c
    return bpc(tt)
ok_u, out_u, in_u = units(0.5, "all"), units(0.5, "out"), units(0.5, "in")
print(f"  half the units removed (W_1 column, bias, W_2 row): {ok_u:.4f}; W_2 row only: {out_u:.4f}; W_1 column only, bias left: {in_u:.4f}")
check(abs(ok_u - out_u) < 1e-9 and abs(in_u - ok_u) > 1e-3, "a unit is silenced by its output row; zeroing only its input column leaves gelu(bias) x row added to every token")

print("Part 4. breakage: a mask that is not re-applied")
def train(reapply, steps=10):
    st = load(CK)
    for v in st.t.values(): v.requires_grad_(True)
    masks = {}
    with torch.no_grad():
        for k in mats:
            w = st.t[k]; idx = w.abs().argsort(0)[:int(w.shape[0] * 0.8)]; m = torch.ones_like(w); m.scatter_(0, idx, 0.0); masks[k] = m; w.mul_(m)
    opt = Adam(3e-4); stream = D.ShardStream(shards(), window=129, batch=8, buffer=256, seed=SEED)
    for s in range(steps):
        X = torch.tensor(stream.next_batch())
        for v in st.t.values(): v.grad = None
        st.loss(X).backward(); opt.step(st.t, 1.0)
        if reapply:
            with torch.no_grad():
                for k, m in masks.items(): st.t[k].mul_(m)
    return sum(int(((st.t[k].detach() != 0) & (m == 0)).sum()) for k, m in masks.items()), sum(int((m == 0).sum()) for m in masks.values())
leak, npruned = train(False); held, _ = train(True)
print(f"  80% pruned, 10 Adam steps: {leak:,} of {npruned:,} pruned weights nonzero without re-masking; {held} with re-masking")
check(leak > 0.9 * npruned and held == 0, "an optimiser step refills pruned weights unless the mask is applied after every step")

print("Part 5. the recorded runs")
J = {os.path.basename(f): json.load(open(f)) for f in glob.glob(os.path.join(HERE, "outputs", "prune_distill_*_500.json"))}
J = {n: j for n, j in J.items() if j["prune"] != "none"}
for n, j in sorted(J.items()): print(f"  {j['prune']:8s} {j['mode']:5s} after pruning {j['after_prune']:.4f} -> {j['final']:.4f}; zeros kept {j['pruned_positions_still_zero']}")
ok = len(J) == 6 and all(j["pruned_positions_still_zero"] and j["final"] < j["after_prune"] for j in J.values())
check(ok, "six prune-then-recover runs recorded; every one recovered and kept its pruned weights at zero")

print("Part 6. the key")
key = None
for line in open(os.path.join(HERE, "..", "..", ".env")):
    if line.startswith("ANTHROPIC_API_KEY="): key = line.split("=", 1)[1].strip().strip('"')
found = [f for f in glob.glob(os.path.join(HERE, "**", "*"), recursive=True) if os.path.isfile(f) and key and key.encode() in open(f, "rb").read()]
check(bool(key) and not found, "the API key was read from .env and its string appears in no file under 04-code/ch34")
print(f"\n{sum(res)} of {len(res)} checkpoints pass")
if not all(res): raise SystemExit(1)
