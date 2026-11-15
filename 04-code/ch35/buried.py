"""Chapter 35, the running sentence buried in the novel: "the cat sat on the" placed after 0 to 2,000 tokens of
Through the Looking-Glass (the filler ends at a sentence end).  For GPT-2 small (up to 1,000 tokens) and SmolLM2-135M
(up to 2,000) record: the log-probability of " mat" after "the cat sat on the"; the same with "dog" in place of "cat"
(how much "cat" matters); the attention from the last " the" (the position that predicts " mat") to " cat", as the largest
share any head gives it and the mean over all heads; and the mean share every head puts on the first token.
Hugging Face reference implementations with eager attention.  CPU, one thread, 32-bit."""
from _common import *
import torch, re
torch.set_num_threads(1); torch.manual_seed(SEED)
from transformers import AutoTokenizer, AutoModelForCausalLM
text = glass_text()
R = {}
for name, lens in (("gpt2", [0, 100, 250, 500, 1000]), ("smollm2", [0, 100, 250, 500, 1000, 2000])):
    tk = AutoTokenizer.from_pretrained(CKPT(name))
    m = AutoModelForCausalLM.from_pretrained(CKPT(name), dtype=torch.float32, attn_implementation="eager").eval()
    ends = [mm.end() for mm in re.finditer(r"[.!?]\s", text)]
    print(f"{name}: filler tokens, log p(' mat') after 'the cat sat on the' | with 'dog' | attention last ' the' -> ' cat': max head, mean of heads | mean share on token 0", flush=True)
    R[name] = {}
    for L in lens:
        # filler: the shortest run of whole sentences from the start of the novel with at least L tokens
        filler = ""
        if L:
            for e in ends:
                if len(tk.encode(text[:e])) >= L: filler = text[:e]; break
        out = {}
        for animal in ("cat", "dog"):
            s = filler + ("The " if not filler else "the ") + f"{animal} sat on the"
            ids = tk.encode(s)
            if name == "smollm2" and False: pass
            with torch.no_grad():
                o = m(torch.tensor([ids]), output_attentions=True)
            lp = torch.log_softmax(o.logits[0, -1].double(), -1)
            mat = tk.encode(" mat")[0]
            out[animal] = float(lp[mat])
            if animal == "cat":
                # position of " cat" (or "cat"): last occurrence among the final 5 tokens
                pos = [i for i in range(len(ids) - 5, len(ids)) if tk.decode([ids[i]]).strip() == "cat"][-1]
                A = torch.stack([a[0, :, -1, :] for a in o.attentions])      # layers x heads x keys, last query row
                out["n"] = len(ids); out["cat max"] = float(A[:, :, pos].max()); out["cat mean"] = float(A[:, :, pos].mean())
                out["first mean"] = float(A[:, :, 0].mean()) if len(ids) > 6 else None
                out["cat by layer max"] = [float(x) for x in A[:, :, pos].max(1).values]
        R[name][L] = out
        fm = "   -  " if out["first mean"] is None else f"{out['first mean']:.3f}"
        print(f"  {L:5d} ({out['n']:5d} tokens)  cat {out['cat']:7.3f}   dog {out['dog']:7.3f}   cat max {out['cat max']:.3f}  mean {out['cat mean']:.4f}   token 0 {fm}", flush=True)
    del m
save_json("buried", R)
