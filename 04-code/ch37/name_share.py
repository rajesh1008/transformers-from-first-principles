"""Chapter 37, 37.6: where the confidence of a greedy answer sits.  SmolLM2-135M and GPT-2 small, "The capital of X is",
greedy 10 tokens (as seen.py).  For each answer: the mean share over all ten tokens (seen.py's number) against the share on
the first token of the first capitalised answer word (the name itself, outside the stop list of agree2.py).
CPU, one thread, 32-bit, seed 1865."""
from _common import *
import torch, numpy as np, re
torch.set_num_threads(1); torch.manual_seed(SEED)
from transformers import AutoTokenizer, AutoModelForCausalLM
from seen_lists import WELL, LESS
from agree2 import STOP, FAKE
R = {}
for name, lab in (("gpt2", "GPT-2 small"), ("smollm2", "SmolLM2-135M")):
    tk = AutoTokenizer.from_pretrained(CKPT(name)); m = AutoModelForCausalLM.from_pretrained(CKPT(name), dtype=torch.float32).eval()
    rows = []
    for grp, items in (("well", WELL), ("less", LESS), ("invented", [(f, None) for f in FAKE])):
        for cn, city in items:
            ids = torch.tensor([tk.encode(f"The capital of {cn} is")])
            with torch.no_grad(): o = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=10, do_sample=False, output_scores=True, return_dict_in_generate=True, pad_token_id=tk.eos_token_id)
            new = [int(t) for t in o.sequences[0, ids.shape[1]:]]
            sh = [float(torch.softmax(sc[0].double(), -1)[t]) for sc, t in zip(o.scores, new)]
            name_sh = None; word = None
            for i, t in enumerate(new):
                s = tk.decode([t]).strip()
                if s[:1].isupper() and re.match(r"[A-Z]", s) and s.rstrip(".,") not in STOP:
                    name_sh = sh[i]; word = s; break
            text = tk.decode(new).replace("\n", " ").strip()
            ok = city is not None and city.lower()[:5] in text.lower()
            rows.append(dict(group=grp, country=cn, text=text, right=bool(ok), mean_share=float(np.mean(sh)), name_token=word, name_share=name_sh))
            print(f"{lab:13s} {cn:13s} {'right' if ok else 'wrong':5s} mean {np.mean(sh):.3f}  name token {word!r} {name_sh if name_sh is None else round(name_sh, 3)}", flush=True)
    R[lab] = rows
    for grp in ("well", "less", "invented"):
        for ok in (True, False):
            v = [r["name_share"] for r in rows if r["group"] == grp and r["right"] == ok and r["name_share"] is not None]
            if v: print(f"{lab} {grp} {'right' if ok else 'not right'}: {len(v)} answers with a name; mean share on the name token {np.mean(v):.3f}", flush=True)
save_json("name_share", R)
