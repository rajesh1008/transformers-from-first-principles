"""Chapter 50 exercises 7 to 11 (added 16 November 2026), with the models and settings of lab50.py: transformers' reference
implementations in 32-bit, eager attention, CPU, one thread, seed 1865.
  50.7  entropy (nats) of every head's attention row for ' mat' (the last of the six pieces), by layer, for the three models
  50.8  a newline before the running sentence: share of attention on the new first token and on 'the', rows 1 onwards
  50.9  Qwen3-0.6B's strongest ' mat'-to-' cat' head (heads50.json) zeroed at the input of its layer's output table: top five
  50.10 64 new tokens from the running sentence, greedy and top-p 0.9 (temperature 1, no top-k) at seeds 1, 2 and 3: the number
        of new tokens before some four-token sequence appears for the second time
  50.11 bits per character on all of The Time Machine (04-code/ch48/corpus/heldout_timemachine.txt), each model with its own
        tokenizer, in non-overlapping windows of 256 predicted tokens with the window's first token as context only (as eval48.py)
-> outputs/ex50.json, logs/ex50.txt"""
import os, json, math, time, torch
torch.set_num_threads(1); torch.manual_seed(1865)
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__))
M = [("GPT-2 small", os.path.join(HERE, "..", "ch29", "ckpt", "gpt2")), ("SmolLM2-135M", os.path.join(HERE, "..", "ch29", "ckpt", "smollm2")),
     ("Qwen3-0.6B", os.path.join(HERE, "ckpt", "qwen3"))]
S = "the cat sat on the mat"; TM = open(os.path.join(HERE, "..", "ch48", "corpus", "heldout_timemachine.txt")).read()
HEADS = {r["model"]: r for r in json.load(open(os.path.join(HERE, "outputs", "heads50.json")))}
def first_repeat(ids, n=4):
    seen = set()
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        if g in seen: return i + n
        seen.add(g)
    return None
out = json.load(open(os.path.join(HERE, "outputs", "ex50.json"))) if os.path.exists(os.path.join(HERE, "outputs", "ex50.json")) else []
def save_stage():
    json.dump(out, open(os.path.join(HERE, "outputs", "ex50.json"), "w"), indent=1, ensure_ascii=False)
for name, path in M:
    if any(r["model"] == name and "timemachine" in r for r in out): continue
    out = [r for r in out if r["model"] != name]
    t0 = time.time(); tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").eval()
    ids = tok(S, add_special_tokens=False)["input_ids"]; r = dict(model=name); out.append(r)
    with torch.no_grad(): A = torch.stack([a[0] for a in m(torch.tensor([ids]), output_attentions=True).attentions])   # [L, H, T, T]
    row = A[:, :, -1, :]; ent = -(row * (row + 1e-12).log()).sum(-1)                                                  # [L, H]
    r["entropy"] = [[round(float(v), 4) for v in l] for l in ent]; L, H = ent.shape
    k = max(1, round(0.1 * L * H)); top = torch.topk(ent.flatten(), k).indices
    r["top10pct_layers"] = sorted(int(i) // H for i in top); r["entropy_uniform_max"] = round(math.log(len(ids)), 4)
    print(f"{name} 50.7: mean entropy {float(ent.mean()):.4f} nats (uniform over six pieces {math.log(6):.4f}); highest tenth of heads ({k}) in layers {r['top10pct_layers']}", flush=True)
    nl = tok("\n" + S, add_special_tokens=False)["input_ids"]
    with torch.no_grad(): A2 = torch.stack([a[0] for a in m(torch.tensor([nl]), output_attentions=True).attentions])
    r["newline_pieces"] = [tok.decode([i]) for i in nl]
    r["newline_first_share"] = round(float(A2[:, :, 1:, 0].mean()), 4); r["newline_the_share"] = round(float(A2[:, :, 2:, 1].mean()), 4)
    r["plain_first_share"] = round(float(A[:, :, 1:, 0].mean()), 4)
    print(f"{name} 50.8: pieces {r['newline_pieces']}; share on the newline {r['newline_first_share']:.4f}, on 'the' {r['newline_the_share']:.4f}"
          f" (without the newline, 'the' as first token took {r['plain_first_share']:.4f})", flush=True)
    save_stage()
    def top5(model_ids):
        with torch.no_grad(): p = torch.softmax(m(torch.tensor([model_ids])).logits[0, -1], -1)
        t = torch.topk(p, 5); return [(tok.decode([int(i)]), round(float(v), 4)) for v, i in zip(t.values, t.indices)], p
    if name == "Qwen3-0.6B":
        hb = HEADS[name]; l, h = hb["best_layer"], hb["best_head"]; hd = m.config.head_dim
        base, p0 = top5(ids)
        hk = m.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda mod, a: (a[0].index_fill(-1, torch.arange(h * hd, (h + 1) * hd), 0.0),))
        abl, p1 = top5(ids); hk.remove()
        r["ablation"] = dict(layer=l, head=h, before=base, after=abl, tv_distance=round(float(0.5 * (p0 - p1).abs().sum()), 4))
        print(f"{name} 50.9: layer {l} head {h} zeroed; before {base}; after {abl}; total-variation distance {r['ablation']['tv_distance']}", flush=True)
    save_stage()
    gen = {}
    with torch.no_grad():
        g = m.generate(torch.tensor([ids]), max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)[0, len(ids):].tolist()
        gen["greedy"] = dict(first_repeat=first_repeat(g), text=tok.decode(g))
        for sd in (1, 2, 3):
            torch.manual_seed(sd)
            g = m.generate(torch.tensor([ids]), max_new_tokens=64, do_sample=True, top_p=0.9, top_k=0, temperature=1.0, pad_token_id=tok.eos_token_id)[0, len(ids):].tolist()
            gen[f"top_p_seed{sd}"] = dict(first_repeat=first_repeat(g), new_tokens=len(g), text=tok.decode(g))
    r["generation"] = gen; save_stage()
    print(f"{name} 50.10: tokens before a four-token sequence repeats (None = not within 64): " + ", ".join(f"{k} {v['first_repeat']}" for k, v in gen.items()), flush=True)
    a = tok(TM, add_special_tokens=False)["input_ids"]; CTX = 256; kw = (len(a) - 1) // CTX
    w = torch.tensor([a[i * CTX: i * CTX + CTX + 1] for i in range(kw)]); chars = sum(len(tok.decode(x[1:].tolist())) for x in w); nats = 0.0
    with torch.no_grad():
        for i in range(0, kw):
            b = w[i:i + 1]; lg = m(b[:, :-1]).logits
            nats += float(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1), reduction="sum"))
    r["timemachine"] = dict(tokens=len(a), windows=kw, chars=chars, loss=round(nats / (kw * CTX), 4), bpc=round(nats / math.log(2) / chars, 4))
    print(f"{name} 50.11: The Time Machine, {len(a):,} tokens, {kw} windows, {chars:,} characters: {r['timemachine']['loss']:.4f} nats per token, {r['timemachine']['bpc']:.4f} bits per character", flush=True)
    r["seconds"] = round(time.time() - t0, 1); save_stage(); del m
json.dump(out, open(os.path.join(HERE, "outputs", "ex50.json"), "w"), indent=1, ensure_ascii=False)
print("done")
