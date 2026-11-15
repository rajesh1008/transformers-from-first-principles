"""Chapter 37, 37.6: agreement between samples against correctness.  GPT-2 small and SmolLM2-135M, "The capital of X is"
for the 30 real countries of seen_lists.py and the 6 invented ones of seen.py: ten continuations of 10 tokens per
question, sampled at temperature 1.0 and at 0.7 (seed 1865 for every question).  Each continuation's answer is its first
capitalised word outside a short stop list ("(none)" if there is none), compared on its first five letters.  Agreement is
the share of the ten taken by the commonest answer other than "(none)".  The majority answer is right if it is the capital.
CPU, one thread, 32-bit.  (agree.py, kept, is the first and cruder version at temperature 1 for SmolLM2 only.)"""
from _common import *
import torch, numpy as np, re, collections
torch.set_num_threads(1)
from transformers import AutoTokenizer, AutoModelForCausalLM
from seen_lists import WELL, LESS
FAKE = ["Zubrovia", "Veltrania", "Oskaria", "Marundi Kel", "Talvessa", "Quorvath"]
STOP = set("The It In This A An City Capital Republic Democratic United State States Kingdom Empire Country He She There I We They "
           "However Its His Her Their Province Region North South East West Central New Old Saint St".split())
def answer(t):
    for w in re.findall(r"\b([A-Z][\w'\-ăâîşţéèü]+)", t):
        if w not in STOP: return w[:5].lower()
    return "(none)"
if __name__ == "__main__":
    R = {}
    for name, lab in (("gpt2", "GPT-2 small"), ("smollm2", "SmolLM2-135M")):
        tk = AutoTokenizer.from_pretrained(CKPT(name)); m = AutoModelForCausalLM.from_pretrained(CKPT(name), dtype=torch.float32).eval()
        for T in (1.0, 0.7):
            rows = []
            for grp, items in (("well", WELL), ("less", LESS), ("invented", [(f, None) for f in FAKE])):
                for cn, city in items:
                    ids = torch.tensor([tk.encode(f"The capital of {cn} is")] * 10)
                    torch.manual_seed(SEED)
                    with torch.no_grad(): out = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=10, do_sample=True, temperature=T, top_k=0, top_p=1.0, pad_token_id=tk.eos_token_id)
                    texts = [tk.decode(o[ids.shape[1]:]) for o in out]; ans = [answer(t) for t in texts]
                    named = [a for a in ans if a != "(none)"]
                    top, k = (collections.Counter(named).most_common(1)[0] if named else ("(none)", 0))
                    right = sum(city is not None and city.lower()[:5] in t.lower() for t in texts)
                    rows.append(dict(group=grp, country=cn, top=top, agree=k / 10, right=right,
                                     majority_right=bool(city is not None and top == city[:5].lower()), answers=ans))
            R[f"{lab} T={T}"] = rows
            real = [r for r in rows if r["group"] != "invented"]; ag = np.array([r["agree"] for r in real]); rt = np.array([r["right"] for r in real])
            mr = np.array([r["majority_right"] for r in real])
            print(f"{lab}, T={T}: right {int(rt[:15].sum())} of 150 (well-known), {int(rt[15:].sum())} of 150 (less-known); "
                  f"correlation of agreement with right-of-10 over 30 countries {np.corrcoef(ag, rt)[0, 1]:.2f}", flush=True)
            for lo, hi in ((0.0, 0.3), (0.3, 0.6), (0.6, 1.01)):
                s = (ag >= lo) & (ag < hi)
                print(f"   agreement {lo:.1f}-{min(hi, 1.0):.1f}: {int(s.sum()):2d} countries, majority answer right in {int(mr[s].sum())}", flush=True)
            wrong_agree = [(r["country"], r["top"], r["agree"]) for r in real if r["agree"] >= 0.5 and not r["majority_right"]]
            print(f"   agreement 0.5 or more with a wrong majority answer: {wrong_agree}", flush=True)
            inv = [(r["country"], r["top"], r["agree"]) for r in rows if r["group"] == "invented"]
            print(f"   invented countries (commonest answer, agreement): {inv}", flush=True)
    save_json("agree2", R)
