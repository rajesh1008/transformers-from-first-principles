"""Chapter 27, 27.11: the invented-word check applied to held-out Carroll text (the same 20 x 100-token stretches as
experiments.py Part K), so the sampling rules' shares have a reference."""
from _common import *
import re
from _pool import corpora
from tiny.eval import windows
t = tok(); c = corpora()
pool_words = set(w.lower() for w in re.findall(r"[A-Za-z]+", t.decode([int(i) for i in c["pool"]])))
W = windows(np.asarray(c["held_b"]), 129)[:20]
texts = [t.decode([int(i) for i in w[1:101]]) for w in W]
ws = [w.lower() for x in texts for w in re.findall(r"[A-Za-z]+", x)]
print(f"held-out 20 x 100 tokens: {len(ws)} words, not in the pool {sum(w not in pool_words for w in ws) / len(ws):.3f}")
