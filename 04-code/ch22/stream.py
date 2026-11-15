"""Chapter 22, s22-8: the Carroll pool as uint16 token shards on disk, read through memory maps by a resumable
stream with a shuffle buffer.  Part 1 writes the shards.  Part 2 measures how fast batches come out (one thread,
fastest of five).  Part 3 stops a stream at batch 200 and resumes it three ways.  Part 4 measures what the shuffle
buffer's size does to a batch: how many of its 8 windows come from the same book, and how far apart in the pool
consecutive windows are.  Shards go to a scratch folder outside the code tree (they are rebuilt in a second)."""
import _common
import os, json, time, shutil, tempfile
import numpy as np
from tiny import data as D
from _common import SEED, save_json
from _pool import corpora, POOL_BOOKS
c = corpora()
out = {}
tmp = os.path.join(tempfile.gettempdir(), "tfp_ch22_shards")
shutil.rmtree(tmp, ignore_errors=True)
docs = [c["books"][b] for b in POOL_BOOKS]
para_docs = []
for b in POOL_BOOKS:                      # documents: each book cut into pieces of at most 8,192 tokens (whole books are uneven)
    x = c["books"][b]
    para_docs += [x[i:i + 8192] for i in range(0, len(x), 8192)]
print("Part 1. shards")
idx = D.write_shards(para_docs, tmp, shard_tokens=32768)
sizes = [os.path.getsize(os.path.join(tmp, s["file"])) for s in idx]
print(f"  {len(para_docs)} documents -> {len(idx)} shards of about 32,768 tokens; {sum(s['tokens'] for s in idx):,} tokens, {sum(sizes):,} bytes on disk")
print(f"  2 bytes per token (uint16; Tiny's ids are below 2,048); the same ids as 64-bit integers in memory: {8 * sum(s['tokens'] for s in idx):,} bytes")
out["shards"] = len(idx); out["bytes"] = sum(sizes)

print("\nPart 2. batches per second (8 windows of 129), one thread, fastest of five runs of 400 batches")
def run_stream(buffer):
    st = D.ShardStream(tmp, buffer=buffer, seed=SEED)
    for _ in range(400):
        st.next_batch()
W = np.stack([c["pool"][i * 128:i * 128 + 129] for i in range(len(c["pool"]) // 128 - 1)])
rng = np.random.default_rng(SEED)
def run_memory():
    for _ in range(400):
        W[rng.integers(0, len(W), 8)]
for name, f in [("in memory, random windows", run_memory), ("shards, buffer 256", lambda: run_stream(256))]:
    ms = _common.best_of(f, 5)
    print(f"  {name:28s} {400 / ms * 1000:10,.0f} batches/s ({ms / 400 * 1000:.1f} microseconds per batch)")
    out[name] = 400 / ms * 1000
print("  a Tiny training step takes about 250 ms (Chapter 21), so either is thousands of times faster than the step")

print("\nPart 3. stop at batch 150 of 300 (2,400 windows, less than one pass), then resume")
ref = D.ShardStream(tmp, buffer=256, seed=SEED)
full = [ref.next_batch() for _ in range(300)]
def key(w):
    return w.tobytes()
seen_full = [key(w) for b in full for w in b]
a = D.ShardStream(tmp, buffer=256, seed=SEED)
first = [a.next_batch() for _ in range(150)]
state = json.loads(json.dumps(a.state_dict()))           # through JSON, as a checkpoint file would be
print(f"  the stream's state: pass {state['passes']}, shard {state['pos']} of {len(idx)}, offset {state['offset']:,}, "
      f"{len(state['buf'])} windows in the buffer, {len(json.dumps(state)):,} bytes as JSON")
def compare(name, rest):
    got = first + rest
    same = sum(int(np.array_equal(x, y)) for x, y in zip(got, full))
    keys = [key(w) for b in got for w in b]
    rep = len(keys) - len(set(keys))
    miss = len(set(seen_full) - set(keys))
    print(f"  {name:44s} batches identical to the uninterrupted run: {same} of 300; windows read twice {rep}; windows skipped {miss}")
    return dict(same=same, repeated=rep, missed=miss)
b = D.ShardStream(tmp, buffer=256, seed=SEED); b.load_state_dict(state)
out["resume_full"] = compare("full state restored", [b.next_batch() for _ in range(150)])
b = D.ShardStream(tmp, buffer=256, seed=SEED)
out["resume_restart"] = compare("restarted from the beginning (no state)", [b.next_batch() for _ in range(150)])
b = D.ShardStream(tmp, buffer=256, seed=SEED)
s2 = dict(state); s2["buf"] = []
b.load_state_dict(s2)
out["resume_nobuffer"] = compare("position restored, buffer not saved", [b.next_batch() for _ in range(150)])

print("\nPart 4. what the shuffle buffer's size does to a batch (400 batches)")
bounds = np.cumsum([0] + [s["tokens"] for s in idx])
book_starts = np.cumsum([0] + [len(c["books"][b]) for b in POOL_BOOKS])
pool_pos = {}
_, maps = D.open_shards(tmp)
for k, m in enumerate(maps):
    for off in range(0, len(m) - 129 + 1, 128):
        pool_pos.setdefault(np.asarray(m[off:off + 129], dtype=np.int64).tobytes(), int(bounds[k] + off))
for buf in (1, 8, 64, 256, 1024):
    st = D.ShardStream(tmp, buffer=buf, seed=SEED)
    same_book, gaps, prev = [], [], None
    for _ in range(400):
        bt = st.next_batch()
        pos = [pool_pos[w.tobytes()] for w in bt]
        bk = [int(np.searchsorted(book_starts, p, side="right") - 1) for p in pos]
        same_book.append(max(np.bincount(bk)))
        for p in pos:
            if prev is not None and p >= 0 and prev >= 0:
                gaps.append(abs(p - prev))
            prev = p
    print(f"  buffer {buf:5d}: largest share of one book in a batch, average {np.mean(same_book) / 8:.2f}; "
          f"median distance between consecutive windows {int(np.median(gaps)):,} tokens")
    out[f"buffer_{buf}"] = dict(same_book=float(np.mean(same_book) / 8), gap=float(np.median(gaps)))
print(f"  (the pool's books are {', '.join(f'{len(c['books'][b]):,}' for b in POOL_BOOKS)} tokens; a window is 128)")
save_json("stream", out)
