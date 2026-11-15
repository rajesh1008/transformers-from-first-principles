"""Chapter 37, 37.2: which of GPT-2 small's near-certain predictions on Through the Looking-Glass are wrong?  The same 40
windows of 513 tokens as calibration.py; for predictions with top share above 0.9 that miss, count the commonest
(predicted, actual) pairs.  CPU, one thread, 32-bit."""
from _common import *
import torch, collections
torch.set_num_threads(1)
from transformers import AutoTokenizer, AutoModelForCausalLM
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); m = AutoModelForCausalLM.from_pretrained(CKPT("gpt2"), dtype=torch.float32).eval()
ids = tk.encode(glass_text()); cnt = collections.Counter(); n_sure = n_wrong = 0
for w in range(40):
    ch = ids[w * 512: w * 512 + 513]
    with torch.no_grad(): p = torch.softmax(m(torch.tensor([ch])).logits[0, :-1].double(), -1)
    conf, top = p.max(-1); tg = torch.tensor(ch[1:])
    for c, t, a in zip(conf.tolist(), top.tolist(), tg.tolist()):
        if c > 0.9:
            n_sure += 1
            if t != a: n_wrong += 1; cnt[(tk.decode([t]), tk.decode([a]))] += 1
print(f"GPT-2 small: {n_sure} predictions above 0.9, {n_wrong} wrong; commonest (predicted, actual):", flush=True)
for (a, b), k in cnt.most_common(12): print(f"   {a!r} -> {b!r}: {k}", flush=True)
save_json("sure_wrong", dict(sure=n_sure, wrong=n_wrong, pairs=[(a, b, k) for (a, b), k in cnt.most_common(30)]))
