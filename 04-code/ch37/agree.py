"""Chapter 37, 37.6: does agreement between samples track correctness?  SmolLM2-135M, "The capital of X is" for the 30 real
countries of seen.py: ten continuations of 10 tokens sampled at temperature 1 (seed 1865).  For each country: how many of
the ten name the real capital, and the agreement of the ten (the share taken by the commonest city-like answer, read as the
first capitalised word after "is").  CPU, one thread, 32-bit."""
from _common import *
import torch, numpy as np, re, collections
torch.set_num_threads(1); torch.manual_seed(SEED)
from transformers import AutoTokenizer, AutoModelForCausalLM
from seen_lists import WELL, LESS
tk = AutoTokenizer.from_pretrained(CKPT("smollm2")); m = AutoModelForCausalLM.from_pretrained(CKPT("smollm2"), dtype=torch.float32).eval()
R = []; g = torch.Generator().manual_seed(SEED)
for cn, city in WELL + LESS:
    ids = torch.tensor([tk.encode(f"The capital of {cn} is")] * 10)
    torch.manual_seed(SEED)
    with torch.no_grad(): out = m.generate(ids, max_new_tokens=10, do_sample=True, temperature=1.0, top_k=0, top_p=1.0, pad_token_id=tk.eos_token_id)
    texts = [tk.decode(o[ids.shape[1]:]) for o in out]
    right = sum(city.lower()[:5] in t.lower() for t in texts)
    names = [(re.findall(r"\b([A-Z][\w'\-ăâîşţ]+)", t) or ["(none)"])[0] for t in texts]
    names = [n for n in names if n not in ("The", "It", "In", "This", "A")] or ["(none)"]
    top, k = collections.Counter(names).most_common(1)[0]
    R.append(dict(country=cn, right=right, agree=k / 10, top=top))
    print(f"{cn:13s} right {right:2d} of 10   commonest capitalised word {top!r} in {k} of 10", flush=True)
ag = np.array([r["agree"] for r in R]); rt = np.array([r["right"] for r in R])
print(f"countries with at least 5 of 10 right: {int((rt >= 5).sum())}; mean agreement there {ag[rt >= 5].mean():.2f}, elsewhere {ag[rt < 5].mean():.2f}", flush=True)
print(f"countries with agreement 0.5 or more and fewer than 5 right: {int(((ag >= 0.5) & (rt < 5)).sum())}", flush=True)
save_json("agree", R)
