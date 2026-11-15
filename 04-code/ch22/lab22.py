"""Lab 22: Tiny's data pipeline, five ways to break it, and a checkpoint.  Seed 1865, one thread.  About a minute.
Stops with an error if a checkpoint fails."""
import _common
import hashlib, json, os, tempfile, shutil
import numpy as np
import torch
torch.set_num_threads(1)
from tiny.model import Tiny
from tiny.torch_tiny import TorchTiny
from tiny import data as D
from _common import novel_ids, tok, SEED
from _pool import corpora, POOL_BOOKS
T = tok(); c = corpora()
def fingerprint(model):
    h = hashlib.sha256()
    for k, v in model.tables().items():
        h.update(k.encode()); h.update(np.ascontiguousarray(v, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]
results = []
def check(name, ok):
    results.append(ok); print(f"  [{'pass' if ok else 'FAIL'}] {name}")
m = Tiny()
tt = TorchTiny.from_numpy(m, dtype=torch.float64)
paras = D.encode_docs(T, D.paragraphs(D.load_source("alice")))

print("Part 1. predict, then read")
w, d = D.pack(paras, 129)
print(f"  the novel's 817 paragraphs packed: {len(w)} windows; the first window holds {len(set(d[0]))} documents")
print(f"  seats of window 0 that may read seat 0 under the document mask: {int(D.doc_mask(d[0])[:, 0].sum())} of 129")
print(f"  chance a pair of similarity 0.5 becomes a candidate with 32 bands of 4: {D.p_candidate(0.5, 32, 4):.3f}; with 16 bands of 8: {D.p_candidate(0.5, 16, 8):.3f}")

print("\nPart 2. five breakages")
# 1. padding counted in the loss
pw, keep = D.pad(paras[:64], 129, pad_id=0)
with torch.no_grad():
    S = tt.scores(torch.tensor(pw[:, :-1]))
    lp = torch.log_softmax(S, -1).gather(-1, torch.tensor(pw[:, 1:])[..., None]).squeeze(-1)
k = torch.tensor(keep[:, 1:])
print(f"  1. padding counted: 64 paragraphs in {len(pw)} padded windows, {1 - k.float().mean():.1%} of targets are padding; "
      f"untrained loss {float(-lp.mean()):.4f} with padding counted, {float(-lp[k].mean()):.4f} on real targets only")
# 2. cross-document context
ws, ds = w[:12], d[:12]
plain = [D.packed_loss(tt, a, b, masked=False) for a, b in zip(ws, ds)]
mask = [D.packed_loss(tt, a, b, masked=True) for a, b in zip(ws, ds)]
nomask_pos = [D.packed_loss(tt, a, b, masked=True, restart_positions=False) for a, b in zip(ws, ds)]
# the same predictions computed one document at a time
sep_tot, sep_n = 0.0, 0
for a, b in zip(ws, ds):
    for doc in np.unique(b):
        seg = a[b == doc]
        if len(seg) > 1:
            with torch.no_grad():
                sep_tot += float(tt.loss(torch.tensor(seg))) * (len(seg) - 1); sep_n += len(seg) - 1
avg = lambda r: sum(x * n for x, n in r) / sum(n for _, n in r)
print(f"  2. cross-document context, first 12 packed windows ({sep_n:,} predictions inside documents):")
print(f"     plain causal mask {avg(plain):.6f} | document mask, positions continue {avg(nomask_pos):.6f} | document mask, positions restart {avg(mask):.6f} | documents read one at a time {sep_tot / sep_n:.6f}")
# 3. contamination
k_raw, n_raw = D.ngram_overlap(D.load_source("alice", raw=True), D.load_source("glass", raw=True))
k_cl, n_cl = D.ngram_overlap(D.load_source("alice"), D.load_source("glass"))
print(f"  3. held-out contamination (data.py's normalisation keeps digits): distinct 13-word runs of Looking-Glass also in Alice, raw files {k_raw:,} of {n_raw:,}; licence removed {k_cl} of {n_cl:,}")
# 4 and 5: the stream
tmp = os.path.join(tempfile.gettempdir(), "tfp_lab22_shards"); shutil.rmtree(tmp, ignore_errors=True)
docs = []
for b in POOL_BOOKS:
    x = c["books"][b]; docs += [x[i:i + 8192] for i in range(0, len(x), 8192)]
D.write_shards(docs, tmp, 32768)
ref = D.ShardStream(tmp, seed=SEED); full = [ref.next_batch() for _ in range(300)]
a = D.ShardStream(tmp, seed=SEED); first = [a.next_batch() for _ in range(150)]
st = json.loads(json.dumps(a.state_dict()))
b = D.ShardStream(tmp, seed=SEED)
rest = [b.next_batch() for _ in range(150)]
keys = [x.tobytes() for bt in first + rest for x in bt]
print(f"  4. a resume that restarts the stream: {len(keys) - len(set(keys)):,} of {len(keys):,} windows read twice in 300 batches")
s1 = D.ShardStream(tmp, buffer=1, seed=SEED)
b1 = [s1.next_batch() for _ in range(50)]
adj = np.mean([np.array_equal(bt[i][-1:], bt[i + 1][:1]) for bt in b1 for i in range(7)])
print(f"  5. a shuffle buffer of 1: {adj:.0%} of neighbouring windows in a batch continue each other (the last token of one is the first of the next)")

print("\nPart 3. checkpoints")
lt = float(m.loss(novel_ids(T)[:1025]))
check(f"tiny/data.py changes nothing about Tiny: {m.n_params:,} learned numbers, fingerprint {fingerprint(m)}, loss {lt:.4f}",
      m.n_params == 2766592 and fingerprint(m) == "19c8710677873d01" and round(lt, 4) == 7.6334)
ids = np.concatenate(paras); tg = np.zeros(len(ids), int)
for r in range(len(w)):
    tg[r * 128 + 1:r * 128 + 129] += 1
check(f"pack() makes every token after the first a target exactly once (up to the dropped tail of {len(ids) - 1 - 128 * len(w)} tokens)",
      set(tg[1:128 * len(w) + 1].tolist()) == {1})
check(f"a packed window with the document mask and restarted positions equals its documents read apart: difference {abs(avg(mask) - sep_tot / sep_n):.1e}",
      abs(avg(mask) - sep_tot / sep_n) < 1e-10)
sa, sb = D.shingles(" ".join(["w%d" % i for i in range(200)])), D.shingles(" ".join(["w%d" % i for i in range(40, 240)]))
J = D.jaccard(sa, sb); est = float(np.mean(D.minhash(sa, 256) == D.minhash(sb, 256)))
check(f"MinHash with 256 numbers estimates a Jaccard similarity of {J:.3f} as {est:.3f}", abs(est - J) < 0.1)
check(f"the banding rule: 32 bands of 4 give {D.p_candidate(0.5, 32, 4):.3f} at similarity 0.5", round(D.p_candidate(0.5, 32, 4), 3) == 0.873)
b = D.ShardStream(tmp, seed=SEED); b.load_state_dict(st)
same = sum(int(np.array_equal(x, y)) for x, y in zip(first + [b.next_batch() for _ in range(150)], full))
check(f"a stream stopped at batch 150 and restored from its state serves the uninterrupted run's batches: {same} of 300", same == 300)
_, maps = D.open_shards(tmp)
back = np.concatenate([np.asarray(x, dtype=np.int64) for x in maps])
check(f"shards read back through memory maps give the pool's {len(back):,} tokens unchanged", np.array_equal(back, np.concatenate(docs)))
print(f"\n{sum(results)} of {len(results)} checkpoints pass")
assert all(results)
