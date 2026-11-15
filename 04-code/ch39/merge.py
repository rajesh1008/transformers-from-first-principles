"""Chapter 39, 39.8: merging a low-rank adapter into Tiny's tables.  Loads Chapter 24's Tiny and the A and B learnt by
tune.py (METHOD=lora RANK=8 TARGETS=all STEPS=1800 SAVE=lora_r8all_1800.pt), then scores the same 20 pool windows and
answers "What did the cat do?" two ways: (1) adapter kept separate, W + (alpha / r) A B computed on the fly; (2) merged,
the product added into W once and the adapter thrown away.  Reports the largest difference in any score, the losses,
the rank of each learned change, the bytes of the adapter file against a full copy of the tables, and the time of
one forward pass each way (wall times vary from run to run and are not speed claims).  CPU, one thread, 32-bit."""
from _common import *
import time, torch, numpy as np
torch.set_num_threads(1)
from tiny.eval import load
from _pool import corpora
T = tok(); EOT = T.specials["<|endoftext|>"]
base = os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")
ck = torch.load(os.path.join(HERE, "outputs", "ckpt", "lora_r8all_1800.pt")); r, alpha, AB = ck["rank"], ck["alpha"], ck["AB"]
pool = [int(i) for i in corpora()["pool"]]
st = np.random.default_rng(SEED + 7).integers(0, len(pool) - 129, size=20); PW = torch.tensor([pool[s:s + 129] for s in st])
sep = load(base); W0 = {k: v.clone() for k, v in sep.t.items()}
def on_the_fly():
    for k, (A, B) in AB.items(): sep.t[k] = W0[k] + (alpha / r) * (A @ B)
merged = load(base)
for k, (A, B) in AB.items(): merged.t[k] = merged.t[k] + (alpha / r) * (A @ B)
def answer(tt, prompt, n=24, hook=None):
    ids = T.encode(f"User: {prompt}\nTiny:"); out = []
    with torch.no_grad():
        for _ in range(n):
            if hook: hook()
            t = int(tt.scores(ids[-128:])[-1].argmax())
            if t == EOT: break
            out.append(t); ids = ids + [t]
    return T.decode(out).strip()
with torch.no_grad():
    t0 = time.perf_counter(); on_the_fly(); s1 = sep.scores(PW[:, :-1]); t1 = time.perf_counter()
    s2 = merged.scores(PW[:, :-1]); t2 = time.perf_counter()
    d = float((s1 - s2).abs().max()); l1, l2 = float(sep.loss(PW)), float(merged.loss(PW))
    base_tt = load(base); l0 = float(base_tt.loss(PW))
print(f"largest difference in any of {s1.numel():,} scores, separate against merged: {d:.2e}")
print(f"pool loss: base {l0:.4f}  adapter separate {l1:.4f}  merged {l2:.4f}")
print(f"cat, separate: {answer(sep, 'What did the cat do?', hook=on_the_fly)!r}   merged: {answer(merged, 'What did the cat do?')!r}")
ranks = sorted({int(torch.linalg.matrix_rank((alpha / r) * (A @ B))) for A, B in AB.values()})
na = sum(A.numel() + B.numel() for A, B in AB.values()); nt = sum(v.numel() for v in W0.values())
print(f"{len(AB)} matrices adapted; rank of every learned change: {ranks}")
print(f"adapter {na:,} numbers = {4 * na:,} bytes at 32 bits; full tables {nt:,} numbers = {4 * nt:,} bytes ({nt / na:.1f} times larger)")
print(f"one forward pass over 20 windows: on the fly (including the 72 products) {1000 * (t1 - t0):.0f} ms, merged {1000 * (t2 - t1):.0f} ms")
save_json("merge", dict(maxdiff=d, loss_base=l0, loss_sep=l1, loss_merged=l2, ranks=ranks, adapter_numbers=na, total=nt))
