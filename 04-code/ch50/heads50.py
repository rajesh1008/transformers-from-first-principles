"""Chapter 50 lab, part 2: in each model, the single head that sends the most attention from ' mat' (position 5) to ' cat'
(position 1), and the share of heads in which ' mat' gives more than half its attention to the first token.  Same models and
settings as lab50.py.  -> outputs/heads50.json.  Seed 1865."""
import os, json, torch
torch.set_num_threads(1); torch.manual_seed(1865)
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__))
M = [("GPT-2 small", os.path.join(HERE, "..", "ch29", "ckpt", "gpt2")), ("SmolLM2-135M", os.path.join(HERE, "..", "ch29", "ckpt", "smollm2")),
     ("Qwen3-0.6B", os.path.join(HERE, "ckpt", "qwen3"))]
S = "the cat sat on the mat"; out = []
for name, path in M:
    tok = AutoTokenizer.from_pretrained(path); m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").eval()
    ids = tok(S, add_special_tokens=False)["input_ids"]
    with torch.no_grad(): A = torch.stack([a[0] for a in m(torch.tensor([ids]), output_attentions=True).attentions])  # [L, H, T, T]
    L, H = A.shape[:2]; v = A[:, :, 5, 1]; i = int(v.argmax()); l, h = divmod(i, H)
    sinky = float((A[:, :, 5, 0] > 0.5).float().mean())
    r = dict(model=name, layers=L, heads=H, best_layer=l, best_head=h, mat_to_cat=round(float(v.max()), 4), row=[round(float(x), 4) for x in A[l, h, 5]],
             heads_over_half_on_first=round(sinky, 4), heads_total=L * H)
    out.append(r); print(json.dumps(r, ensure_ascii=False)); del m
json.dump(out, open(os.path.join(HERE, "outputs", "heads50.json"), "w"), indent=1)
