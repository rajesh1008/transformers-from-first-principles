"""Chapter 29, section 29.5: Tiny (Chapter 24's 2,500-step model) on the same held-out text as gpt2_decode.py Part D, in
windows of up to 1,024 tokens like GPT-2's, so that the two bits-per-character figures differ only in the model and tokenizer.
32-bit, one thread."""
from _common import *
import torch, math
torch.set_num_threads(1)
from tiny.eval import load
from _pool import corpora
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
t = tok(); c = corpora(); ids = [int(i) for i in c["held_b"]]
for ctx in (129, 1024):
    tot, chars, n = 0.0, 0, 0
    for a in range(0, len(ids) - 1, ctx):
        w = ids[a:a + ctx]
        if len(w) < 2: break
        with torch.no_grad():
            lp = torch.log_softmax(tt.scores(w[:-1]).double(), -1)
        s = -lp[torch.arange(len(w) - 1), torch.tensor(w[1:])]
        tot += float(s.sum()); n += len(w) - 1; chars += len(t.decode(w[1:]))
    print(f"Tiny, windows of {ctx:4d}: mean surprise {tot / n:.4f} nats per token over {n:,} tokens; {tot / math.log(2) / chars:.4f} bits per character")
print("Tiny, windows of 1024, mean surprise by position in the window (Tiny trained on windows of 129 tokens, Chapter 24):")
buckets = {}
for a in range(0, len(ids) - 1, 1024):
    w = ids[a:a + 1024]
    with torch.no_grad():
        lp = torch.log_softmax(tt.scores(w[:-1]).double(), -1)
    s = (-lp[torch.arange(len(w) - 1), torch.tensor(w[1:])]).numpy()
    for lo, hi in [(0, 128), (128, 256), (256, 512), (512, 1023)]:
        buckets.setdefault((lo, hi), []).extend(s[lo:hi].tolist())
for (lo, hi), v in buckets.items():
    print(f"  positions {lo:4d} to {hi - 1:4d}: {len(v):5d} tokens, mean surprise {np.mean(v):.4f}")
