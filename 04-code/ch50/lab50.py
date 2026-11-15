"""Chapter 50 lab: the running sentence through three generations of open model, with transformers' reference implementations
in 32-bit on one thread: GPT-2 small (2019; 04-code/ch29/ckpt/gpt2), SmolLM2-135M (2024; 04-code/ch29/ckpt/smollm2) and
Qwen3-0.6B (2025; ckpt/qwen3, downloaded from Hugging Face Qwen/Qwen3-0.6B, Apache-2.0).  For each: design from the config,
learned numbers, how the sentence is cut, the six-by-six attention table of the last token row averaged over heads in the
first, middle and last layer, the share of attention on the first token (the 'sink') per layer, the five likeliest next
tokens after the sentence, and a greedy 16-token continuation.  -> outputs/lab50.json, logs/lab50.txt.  Seed 1865."""
import os, json, math, time, torch
torch.set_num_threads(1); torch.manual_seed(1865)
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__))
M = [("GPT-2 small", "2019", os.path.join(HERE, "..", "ch29", "ckpt", "gpt2")),
     ("SmolLM2-135M", "2024", os.path.join(HERE, "..", "ch29", "ckpt", "smollm2")),
     ("Qwen3-0.6B", "2025", os.path.join(HERE, "ckpt", "qwen3"))]
S = "the cat sat on the mat"
out = []
for name, year, path in M:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager"); model.eval()
    c = model.config
    ids = tok(S, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        o = model(torch.tensor([ids]), output_attentions=True)
    A = [a[0].float() for a in o.attentions]                     # per layer [heads, T, T]
    T = len(ids); L = len(A)
    sink = [float(a[:, 1:, 0].mean()) for a in A]               # rows 1.. (row 0 can only see itself)
    ent = [float(-(a[:, -1] * (a[:, -1] + 1e-12).log()).sum(-1).mean()) for a in A]
    tables = {lab: [[round(float(v), 4) for v in row] for row in A[i].mean(0)] for lab, i in (("first", 0), ("middle", L // 2), ("last", L - 1))}
    p = torch.softmax(o.logits[0, -1], -1); top = torch.topk(p, 5)
    with torch.no_grad():
        g = model.generate(torch.tensor([ids]), max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
    cont = tok.decode(g[0, T:])
    nparam = sum(x.numel() for x in model.parameters())
    r = dict(model=name, year=year, learned_numbers=nparam, layers=c.num_hidden_layers, width=c.hidden_size,
             heads=c.num_attention_heads, kv_heads=getattr(c, "num_key_value_heads", c.num_attention_heads),
             vocab=c.vocab_size, context=getattr(c, "max_position_embeddings", getattr(c, "n_positions", None)),
             positions="learned" if c.model_type == "gpt2" else f"rotary, theta {json.load(open(os.path.join(path, 'config.json'))).get('rope_theta'):g}",
             norm="layer norm" if c.model_type == "gpt2" else "RMSNorm", ffn=getattr(c, "activation_function", None) or f"gated {c.hidden_act}",
             qk_norm=c.model_type == "qwen3", tied=bool(getattr(c, "tie_word_embeddings", True)),
             pieces=[tok.decode([i]) for i in ids], ids=ids,
             sink_by_layer=[round(s, 4) for s in sink], sink_mean=round(sum(sink) / L, 4),
             last_row_entropy_by_layer=[round(e, 4) for e in ent], tables=tables,
             top5=[(tok.decode([int(i)]), round(float(v), 4)) for v, i in zip(top.values, top.indices)],
             greedy16=cont, seconds=round(time.time() - t0, 1))
    out.append(r); print(json.dumps({k: v for k, v in r.items() if k not in ("tables", "sink_by_layer", "last_row_entropy_by_layer")}, ensure_ascii=False))
    print("  sink by layer", r["sink_by_layer"]); print("  last-layer table", tables["last"][-1])
    del model; import gc; gc.collect()
json.dump(out, open(os.path.join(HERE, "outputs", "lab50.json"), "w"), indent=1, ensure_ascii=False)
chk = [("each model cuts the running sentence into six pieces", all(len(r["ids"]) == 6 for r in out)),
       ("every attention row sums to one (checked on the stored last-layer tables)", all(abs(sum(row) - 1) < 1e-3 for r in out for row in r["tables"]["last"])),
       ("the newer models use rotary positions and RMSNorm; GPT-2 does not", out[0]["positions"] == "learned" and all(r["norm"] == "RMSNorm" for r in out[1:])),
       ("grouped key-value heads in both newer models", all(r["kv_heads"] < r["heads"] for r in out[1:])),
       ("all three put more than a third of attention on the first token on average", all(r["sink_mean"] > 1 / 3 for r in out))]
for i, (lab, ok) in enumerate(chk, 1): print(f"check {i}, {lab}: {'pass' if ok else 'FAIL'}")
print(f"{sum(ok for _, ok in chk)} of {len(chk)} checks pass")
