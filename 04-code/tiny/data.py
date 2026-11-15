"""Tiny's data pipeline (Chapter 22).  Everything a training run needs between a folder of books and a batch of
windows: sources and documents, packing and padding, document masks and positions, exact and near-duplicate
detection (MinHash with banding), uint16 token shards on disk read through a memory map, a resumable stream with a
shuffle buffer, mixtures of sources, and orderings (curricula).  Nothing here changes Tiny itself: the model still has
2,766,592 learned numbers and no separator token, so document boundaries travel beside the tokens, never inside them.
Seed 1865 wherever a random number is drawn."""
import os, re, json, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# ---- sources ----------------------------------------------------------------------------------------------
# Every source is a public-domain Project Gutenberg plain-text file, fetched 31 Oct 2026 (Alice, Looking-Glass)
# or 11 Nov 2026 (the rest).  The licence header and footer are removed before anything else happens.
SOURCES = {
    "alice":    ("alice_gutenberg_raw.txt", "Alice's Adventures in Wonderland (1865), eBook #11"),
    "glass":    ("looking_glass_gutenberg_raw.txt", "Through the Looking-Glass (1871), eBook #12; held out"),
    "sylvie":   ("sylvie_and_bruno_gutenberg_raw.txt", "Sylvie and Bruno (1889), eBook #620"),
    "sylvie2":  ("sylvie_and_bruno_concluded_gutenberg_raw.txt", "Sylvie and Bruno Concluded (1893), eBook #48795"),
    "tangled":  ("tangled_tale_gutenberg_raw.txt", "A Tangled Tale (1885), eBook #29042"),
}

def strip_gutenberg(raw):
    """The text between the START and END markers, without the licence."""
    s = raw.index("*** START OF THE PROJECT GUTENBERG EBOOK")
    s = raw.index("\n", s) + 1
    e = raw.index("*** END OF THE PROJECT GUTENBERG EBOOK")
    return raw[s:e].strip()

def load_source(name, raw=False):
    text = open(os.path.join(DATA, SOURCES[name][0]), encoding="utf-8").read()
    return text if raw else strip_gutenberg(text)

def paragraphs(text):
    """Documents at the paragraph level: blocks separated by a blank line, inner line breaks kept."""
    return [p.strip("\n") for p in re.split(r"\n\s*\n", text) if p.strip()]

def chapters(text, pattern=r"\n(?=(?:CHAPTER|KNOT) [0-9IVXL]+\.)"):
    return [p for p in re.split(pattern, text) if p.strip()]

def encode_docs(tok, docs):
    return [np.asarray(tok.encode(d), dtype=np.int64) for d in docs]

# ---- documents into windows -------------------------------------------------------------------------------
def pack(doc_ids, window=129):
    """Lay documents end to end and cut into non-overlapping windows of `window` tokens that share their edge token
    (window w holds tokens 128w .. 128w + 128, so every token after the first is a target exactly once).
    Returns (windows, docs): two (n, window) int arrays, the token ids and the index of the document each came from.
    The tail that does not fill a window is dropped."""
    ids = np.concatenate(doc_ids)
    doc = np.concatenate([np.full(len(d), i) for i, d in enumerate(doc_ids)])
    step = window - 1
    n = (len(ids) - 1) // step
    idx = np.arange(n)[:, None] * step + np.arange(window)[None, :]
    return ids[idx], doc[idx]

def pad(doc_ids, window=129, pad_id=0):
    """Each document in its own windows.  Returns (windows, keep): keep[i, j] is False where window i's token j is
    padding, so a prediction of seat j is real only when keep[i, j] (the target) is True."""
    rows, keep = [], []
    step = window - 1
    for d in doc_ids:
        for s in range(0, max(len(d) - 1, 1), step):
            w = d[s:s + window]
            k = np.ones(window, bool); k[len(w):] = False
            rows.append(np.concatenate([w, np.full(window - len(w), pad_id)])); keep.append(k)
    return np.array(rows), np.array(keep)

def doc_mask(doc_row):
    """Intra-document causal mask for one packed window: seat j may read seat i only if i <= j and both come
    from the same document."""
    d = np.asarray(doc_row)
    n = len(d)
    return np.tril(np.ones((n, n), bool)) & (d[:, None] == d[None, :])

def doc_positions(doc_row):
    """Positions that restart at 0 at each document boundary inside a packed window."""
    d = np.asarray(doc_row)
    pos = np.zeros(len(d), dtype=np.int64)
    for j in range(1, len(d)):
        pos[j] = 0 if d[j] != d[j - 1] else pos[j - 1] + 1
    return pos

# ---- exact and near-duplicate detection --------------------------------------------------------------------
def normalise(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower().replace("\u2019", "'")))

def exact_key(text):
    """A fingerprint of a document's normalised text: equal keys mean equal text after normalising."""
    return hashlib.blake2b(normalise(text).encode(), digest_size=8).hexdigest()

def shingles(text, k=5):
    """The set of k-word runs of a document, each hashed to a 64-bit integer."""
    w = normalise(text).split()
    if len(w) < k:
        w = w + [""] * (k - len(w))
    return {int.from_bytes(hashlib.blake2b(" ".join(w[i:i + k]).encode(), digest_size=8).digest(), "little")
            for i in range(len(w) - k + 1)}

def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0

MERSENNE = (1 << 61) - 1

def minhash(sh, n_hash=128, seed=1865):
    """MinHash signature: for each of n_hash random functions h(x) = (a x + b) mod p, the smallest h over the set.
    Two sets agree in any one position with probability equal to their Jaccard similarity."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, MERSENNE, size=n_hash, dtype=np.uint64).astype(object)
    b = rng.integers(0, MERSENNE, size=n_hash, dtype=np.uint64).astype(object)
    x = [v % MERSENNE for v in sh]
    return np.array([min((ai * v + bi) % MERSENNE for v in x) for ai, bi in zip(a, b)], dtype=object)

def p_candidate(s, bands, rows):
    """Probability that two documents of Jaccard similarity s share at least one band of `rows` signature numbers."""
    return 1.0 - (1.0 - s ** rows) ** bands

def lsh_pairs(sigs, bands, rows):
    """Candidate pairs: documents whose signatures agree on every number of at least one band."""
    pairs = set()
    for bnd in range(bands):
        buckets = {}
        for i, s in enumerate(sigs):
            key = tuple(s[bnd * rows:(bnd + 1) * rows])
            buckets.setdefault(key, []).append(i)
        for ids in buckets.values():
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    pairs.add((ids[x], ids[y]))
    return pairs

def ngram_overlap(train_text, test_text, n=13):
    """Distinct word n-grams of test_text that also occur in train_text (the contamination check of s22-4)."""
    def g(t):
        w = normalise(t).split()
        return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
    a, b = g(train_text), g(test_text)
    return len(a & b), len(b)

# ---- shards on disk ----------------------------------------------------------------------------------------
def write_shards(doc_ids, outdir, shard_tokens=65536, name="shard"):
    """Write documents end to end as uint16 token files of about shard_tokens tokens each (a document is never
    split across shards), with an index.json listing each shard's file, token count and document starts."""
    os.makedirs(outdir, exist_ok=True)
    index, buf, starts, k = [], [], [], 0
    def flush():
        nonlocal buf, starts, k
        if not buf:
            return
        arr = np.concatenate(buf).astype(np.uint16)
        fn = f"{name}_{k:04d}.bin"
        arr.tofile(os.path.join(outdir, fn))
        index.append(dict(file=fn, tokens=int(len(arr)), doc_starts=starts))
        buf, starts, k = [], [], k + 1
    n = 0
    for d in doc_ids:
        assert d.max() < 65536, "uint16 holds ids below 65,536 only"
        starts.append(int(sum(len(x) for x in buf)))
        buf.append(np.asarray(d))
        n += len(d)
        if sum(len(x) for x in buf) >= shard_tokens:
            flush()
    flush()
    json.dump(index, open(os.path.join(outdir, "index.json"), "w"), indent=1)
    return index

def open_shards(outdir):
    index = json.load(open(os.path.join(outdir, "index.json")))
    return index, [np.memmap(os.path.join(outdir, s["file"]), dtype=np.uint16, mode="r") for s in index]

class ShardStream:
    """Windows of `window` tokens read in order from memory-mapped shards, cut without overlap except the shared
    edge token, passed through a shuffle buffer of `buffer` windows, and served in batches.  Shard order is
    shuffled once per pass with the stream's own generator.  The whole position of the stream (pass, place in the
    shard order, shard offset, the buffer's contents and the generator's state) is in state_dict(), so a run that
    stops at step s and reloads it serves exactly the batches an uninterrupted run would have served from s on."""
    def __init__(self, outdir, window=129, batch=8, buffer=256, seed=1865):
        self.index, self.maps = open_shards(outdir)
        self.window, self.batch, self.buffer = window, batch, buffer
        self.rng = np.random.default_rng(seed)
        self.passes, self.pos, self.offset, self.buf = 0, 0, 0, []
        self.order = self.rng.permutation(len(self.maps)).tolist()
        self.served = 0

    def _next_window(self):
        step = self.window - 1
        while True:
            m = self.maps[self.order[self.pos]]
            if self.offset + self.window <= len(m):
                w = np.asarray(m[self.offset:self.offset + self.window], dtype=np.int64)
                self.offset += step
                return w
            self.pos, self.offset = self.pos + 1, 0
            if self.pos == len(self.order):
                self.passes += 1
                self.pos = 0
                self.order = self.rng.permutation(len(self.maps)).tolist()

    def next_batch(self):
        out = []
        while len(out) < self.batch:
            while len(self.buf) < self.buffer:
                self.buf.append(self._next_window())
            j = int(self.rng.integers(len(self.buf)))
            self.buf[j], self.buf[-1] = self.buf[-1], self.buf[j]
            out.append(self.buf.pop())
        self.served += 1
        return np.stack(out)

    def state_dict(self):
        return dict(passes=self.passes, pos=self.pos, offset=self.offset, order=list(self.order),
                    buf=[w.tolist() for w in self.buf], rng=self.rng.bit_generator.state, served=self.served)

    def load_state_dict(self, s):
        self.passes, self.pos, self.offset, self.order = s["passes"], s["pos"], s["offset"], list(s["order"])
        self.buf = [np.array(w, dtype=np.int64) for w in s["buf"]]
        self.rng.bit_generator.state = s["rng"]
        self.served = s["served"]

# ---- mixtures and orderings --------------------------------------------------------------------------------
def mixture_starts(lengths, weights, n, window=129, seed=1865):
    """n window draws from several token streams: each draw picks a source with probability weights[k], then a
    random start in it.  Returns a list of (source, start)."""
    rng = np.random.default_rng(seed)
    w = np.asarray(weights, float) / np.sum(weights)
    src = rng.choice(len(w), size=n, p=w)
    return [(int(k), int(rng.integers(0, lengths[k] - window))) for k in src]

def fixed_passes(windows, passes, steps, batch=8, seed=1865):
    """The batches of a run that reads `windows` (an (n, window) array) `passes` times in fresh random orders,
    truncated to steps x batch windows.  Returns an (steps, batch, window) array."""
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(len(windows)) for _ in range(passes)])[:steps * batch]
    assert len(order) == steps * batch, "not enough windows for the steps asked"
    return windows[order].reshape(steps, batch, -1)

def unigram_surprise(windows, counts):
    """A model-free difficulty score: the mean surprise (natural log) of each window's targets under the token
    frequencies `counts` of the training data.  Windows of common tokens score low ("easy")."""
    p = (counts + 1) / (counts.sum() + len(counts))
    return -np.log(p)[windows[:, 1:]].mean(1)

# ---- a packed window read with a document mask (needs PyTorch) ---------------------------------------------
def packed_loss(tt, window, doc_row, masked=True, restart_positions=True):
    """The loss of one packed window under TorchTiny tt.  masked=False is the plain causal mask (every seat reads
    every earlier seat, whatever its document).  masked=True uses doc_mask and, if restart_positions, positions that
    restart at each document, so the window's loss is exactly the loss of its documents read as separate windows.
    The prediction that crosses a boundary (last token of one document -> first of the next) is left out in both
    cases.  Returns (mean surprise over the kept predictions, number of kept predictions)."""
    import math, torch
    ids = torch.as_tensor(np.asarray(window), dtype=torch.long)
    d = np.asarray(doc_row)
    n = len(ids) - 1
    if masked:
        M = torch.as_tensor(doc_mask(d[:-1]))
        pos = torch.as_tensor(doc_positions(d[:-1])) if restart_positions else torch.arange(n)
    else:
        M = torch.tril(torch.ones(n, n, dtype=torch.bool)); pos = torch.arange(n)
    x = tt.t["E"][ids[:-1]] + tt.t["P"][pos]
    h = tt.config["n_heads"]
    for l in range(tt.config["n_layers"]):
        xn = tt._norm(x, f"h.{l}.ln_1"); p = f"h.{l}.att."
        sp = lambda T: T.reshape(n, h, -1).transpose(0, 1)
        Q, K, V = sp(xn @ tt.t[p + "W_q"]), sp(xn @ tt.t[p + "W_k"]), sp(xn @ tt.t[p + "W_v"])
        sc = (Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1])).masked_fill(~M, float("-inf"))
        a = (torch.softmax(sc, -1) @ V).transpose(0, 1).reshape(n, -1) @ tt.t[p + "W_o"]
        x = x + a
        x = x + tt._ffn(tt._norm(x, f"h.{l}.ln_2"), l)
    if tt.config["final_norm"]:
        x = tt._norm(x, "ln_f")
    S = x @ (tt.t["E"].T if tt.config["tied"] else tt.t["W_out"])
    lp = torch.log_softmax(S, -1).gather(-1, ids[1:, None]).squeeze(-1)
    keep = torch.as_tensor(d[1:] == d[:-1])
    return float(-lp[keep].mean().detach()), int(keep.sum())
