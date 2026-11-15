"""Chapter 29, section 29.6: Tiny under exactly gpt2_decode.py Part E's design (3 prompts x 5 continuations x 40 tokens, seed
1865), with both invented-word counts, so that the two models' rows can be read side by side.  32-bit, one thread."""
from _common import *
import torch, re
torch.set_num_threads(1)
from tiny.eval import load
from tiny.seed import streams
from tiny import generate as G
from _pool import corpora
from transformers import AutoTokenizer
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
t = tok(); c = corpora(); tk = AutoTokenizer.from_pretrained(os.path.join(HERE, "ckpt", "gpt2"))
pool_words = set(w.lower() for w in re.findall(r"[A-Za-z]+", t.decode([int(i) for i in c["pool"]])))
lex = set(s.strip().lower() for s in (tk.decode([i]) for i in range(tk.vocab_size)) if re.fullmatch(r" [A-Za-z]+", s))
PROMPTS = ["the cat sat on the", "Alice was beginning to", '"I don\'t know," said the']
for label, kw in [("greedy", None), ("T 1.0", dict(T=1.0)), ("T 0.7", dict(T=0.7)), ("min-p 0.1", dict(min_p=0.1)), ("top-p 0.9", dict(top_p=0.9))]:
    rng = None if kw is None else streams(SEED)["sample"]; texts, seqs, su = [], [], []
    for pr in PROMPTS:
        for _ in range(1 if kw is None else 5):
            p0 = t.encode(pr); ids, info = G.generate(tt, p0, 40, rng, **(kw or {}))
            seqs.append(ids[len(p0):]); texts.append(t.decode(ids[len(p0):])); su += info["surprise"]
    ws = [w.lower() for x in texts for w in re.findall(r"[A-Za-z]+", x)]
    a = sum(w not in pool_words for w in ws) / len(ws); b = sum(w not in pool_words and w not in lex for w in ws) / len(ws)
    print(f"  Tiny {label:<10} surprise {np.mean(su):.3f}  rep4 {np.mean([G.repeated_runs(s) for s in seqs]):.3f}  not-in-pool {a:.3f}  "
          f"not-in-pool-or-lexicon {b:.3f}  distinct-2 {G.distinct_n(seqs, 2):.3f}  p1 {texts[0][:60]!r}")
