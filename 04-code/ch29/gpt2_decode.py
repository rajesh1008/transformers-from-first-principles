"""Chapter 29, sections 29.5 and 29.6: GPT-2 small, loaded with load_gpt2.py's forward pass, scored on Chapter 22's held-out
Carroll text in bits per character (to compare with Tiny across two tokenizers), then decoded with Chapters 27 and 28's
rules through tiny/generate.py.  32-bit, one thread, seed 1865."""
import sys, os, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import load_gpt2 as LG                         # reuses the tables and forward pass; its own printout is discarded
from _common import *
import torch, re, math
from tiny.seed import streams
from tiny import generate as G
from _pool import corpora
torch.set_num_threads(1)
tk = LG.tk
class GPT2:
    def scores(self, ids):
        with torch.no_grad():
            return LG.forward(list(ids))
gm = GPT2()
c = corpora(); t = tok()
out = {}
print("Part D: held-out Carroll (Chapter 22's last 4,000 Tiny tokens of Sylvie and Bruno Concluded), bits per character")
text = t.decode([int(i) for i in c["held_b"]])
g_ids = tk.encode(text)
tot, chars, ntok = 0.0, 0, 0
for a in range(0, len(g_ids) - 1, 1024):
    w = g_ids[a:a + 1024]
    if len(w) < 2: break
    with torch.no_grad():
        lp = torch.log_softmax(LG.forward(w).double(), -1)
    s = -lp[torch.arange(len(w) - 1), torch.tensor(w[1:])]
    tot += float(s.sum()); ntok += len(w) - 1; chars += len(tk.decode(w[1:]))
print(f"  text {len(text):,} characters; GPT-2 tokens {len(g_ids):,} ({len(g_ids) / len(text):.3f} per character), Tiny tokens {len(c['held_b']):,} ({len(c['held_b']) / len(text):.3f} per character)")
print(f"  GPT-2: mean surprise {tot / ntok:.4f} nats per token over {ntok:,} predicted tokens; {tot / math.log(2) / chars:.4f} bits per character")
out["bpc"] = dict(gpt2=tot / math.log(2) / chars, loss=tot / ntok, tokens=len(g_ids), chars=len(text))

pool_words = set(w.lower() for w in re.findall(r"[A-Za-z]+", t.decode([int(i) for i in c["pool"]])))
lex = set(s.strip().lower() for s in (tk.decode([i]) for i in range(tk.vocab_size)) if re.fullmatch(r" [A-Za-z]+", s))
PROMPTS = ["the cat sat on the", "Alice was beginning to", '"I don\'t know," said the']
def measures(texts):
    ws = [w.lower() for x in texts for w in re.findall(r"[A-Za-z]+", x)]
    a = sum(w not in pool_words for w in ws) / len(ws)
    b = sum(w not in pool_words and w not in lex for w in ws) / len(ws)
    return a, b
print(f"\nPart E: decoding GPT-2 (3 prompts x 5 continuations x 40 tokens; seed 1865).  'not in pool' as for Tiny; 'not in pool"
      f" or GPT-2's {len(lex):,} whole-word tokens' as a stricter count of invented words")
for label, kw in [("greedy", None), ("T 1.0", dict(T=1.0)), ("T 0.7", dict(T=0.7)), ("min-p 0.1", dict(min_p=0.1)), ("top-p 0.9", dict(top_p=0.9))]:
    rng = None if kw is None else streams(SEED)["sample"]; texts, seqs, su = [], [], []
    for pr in PROMPTS:
        for _ in range(1 if kw is None else 5):
            p0 = tk.encode(pr)
            ids, info = G.generate(gm, p0, 40, rng, **(kw or {}))
            seqs.append(ids[len(p0):]); texts.append(tk.decode(ids[len(p0):])); su += info["surprise"]
    a, b = measures(texts)
    rep = float(np.mean([G.repeated_runs(s_) for s_ in seqs]))
    print(f"  {label:<10} surprise {np.mean(su):.3f}  rep4 {rep:.3f}  not-in-pool {a:.3f}  not-in-pool-or-lexicon {b:.3f}  "
          f"distinct-2 {G.distinct_n(seqs, 2):.3f}  p1 {texts[0][:70]!r}", flush=True)
    out[label] = dict(surprise=float(np.mean(su)), rep4=rep, novel=a, novel_strict=b, d2=G.distinct_n(seqs, 2), first=texts[0])
save_json("gpt2_decode", out)
