"""Chapter 29, section 29.8: a second family.  SmolLM2-135M (HuggingFaceTB/SmolLM2-135M, a Llama-shaped model) loaded from its
released files and run with the book's own forward pass: RMSNorm, rotary positions (half-split pairing, theta 100,000),
grouped key-value heads (9 query heads sharing 3 key-value heads), the SwiGLU feed-forward and a tied output table.  The file
stores BF16; every table is widened to 32-bit and the pass runs in 32-bit, one thread.  Checked against transformers'
LlamaForCausalLM in 32-bit; then four loading mistakes; then bits per character on the held-out Carroll text."""
from _common import *
import torch, json, struct, math
torch.set_num_threads(1)
from safetensors.torch import load_file
D = os.path.join(HERE, "ckpt", "smollm2")
cfg = json.load(open(os.path.join(D, "config.json")))
raw = open(os.path.join(D, "model.safetensors"), "rb").read(); n = struct.unpack("<Q", raw[:8])[0]
head = json.loads(raw[8:8 + n]); head.pop("__metadata__", None)
print(f"Part A: {len(raw):,} bytes; {len(head)} tables; formats {sorted({v['dtype'] for v in head.values()})}; "
      f"numbers {sum(int(np.prod(v['shape'])) for v in head.values()):,}")
for k in sorted(k for k in head if ".layers.0." in k) + ["model.embed_tokens.weight", "model.norm.weight"]:
    print(f"    {k:<44} {head[k]['dtype']} {head[k]['shape']}")
W = {k: v.float() for k, v in load_file(os.path.join(D, "model.safetensors")).items()}
L, d, Hq, Hkv, hd = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"] // cfg["num_attention_heads"]
eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]

def rms(x, w, layer=False):
    if layer:                                                   # breakage: layer norm (mean removed) instead of RMSNorm
        x = x - x.mean(-1, keepdim=True)
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w

def rope(x, pos, th=theta, interleaved=False):
    inv = 1.0 / th ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)
    ang = pos[:, None].float() * inv[None]                      # [T, hd/2]
    c, s = torch.cos(ang), torch.sin(ang)
    if interleaved:                                             # breakage: pair dims (0,1), (2,3), ... instead of (i, i + hd/2)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)

def forward(ids, th=theta, interleaved=False, layer_norm=False, kv_tile=False):
    T = len(ids); pos = torch.arange(T)
    x = W["model.embed_tokens.weight"][torch.tensor(ids)]
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    for l in range(L):
        p = f"model.layers.{l}."
        a = rms(x, W[p + "input_layernorm.weight"], layer_norm)
        q = (a @ W[p + "self_attn.q_proj.weight"].T).view(T, Hq, hd).transpose(0, 1)      # nn.Linear: [out, in]
        k = (a @ W[p + "self_attn.k_proj.weight"].T).view(T, Hkv, hd).transpose(0, 1)
        v = (a @ W[p + "self_attn.v_proj.weight"].T).view(T, Hkv, hd).transpose(0, 1)
        q, k = rope(q, pos, th, interleaved), rope(k, pos, th, interleaved)
        rep = Hq // Hkv
        k, v = (k.repeat(rep, 1, 1), v.repeat(rep, 1, 1)) if kv_tile else (k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0))
        s = (q @ k.transpose(-1, -2) / math.sqrt(hd)).masked_fill(~mask, float("-inf"))
        o = (torch.softmax(s, -1) @ v).transpose(0, 1).reshape(T, d)
        x = x + o @ W[p + "self_attn.o_proj.weight"].T
        m = rms(x, W[p + "post_attention_layernorm.weight"], layer_norm)
        g, u = m @ W[p + "mlp.gate_proj.weight"].T, m @ W[p + "mlp.up_proj.weight"].T
        x = x + (torch.nn.functional.silu(g) * u) @ W[p + "mlp.down_proj.weight"].T
    return rms(x, W["model.norm.weight"], layer_norm) @ W["model.embed_tokens.weight"].T

from transformers import AutoTokenizer, LlamaForCausalLM
tk = AutoTokenizer.from_pretrained(D)
ids = tk.encode(SENTENCE)
ref = LlamaForCausalLM.from_pretrained(D, torch_dtype=torch.float32).eval()
with torch.no_grad():
    R = ref(torch.tensor([ids])).logits[0]; S = forward(ids)
lr = torch.log_softmax(R, -1)
learned = sum(int(np.prod(v["shape"])) for v in head.values())
print(f"\nPart B: running sentence in SmolLM2's tokenizer: {ids} -> {[tk.decode([i]) for i in ids]}")
print(f"  own forward vs reference: largest |difference| in log-probabilities {(torch.log_softmax(S, -1) - lr).abs().max():.3e}; "
      f"same top token everywhere {bool((S.argmax(-1) == R.argmax(-1)).all())}")
pre = tk.encode("the cat sat on the")
with torch.no_grad():
    p = torch.softmax(forward(pre)[-1].double(), -1)
Hn = float(-(p * torch.log(p.clamp_min(1e-300))).sum()); top = torch.argsort(-p)[:10]
mat = tk.encode(" mat")
print(f"  after 'the cat sat on the': entropy {Hn:.3f} nats; top 10: " + ", ".join(f"{tk.decode([int(j)])!r} {float(p[j]):.4f}" for j in top))
print(f"  ' mat' is {len(mat)} token(s) {mat}; first-token share {float(p[mat[0]]):.4f}, rank {int((p > p[mat[0]]).sum()) + 1}")

print("\nPart C: four ways to load it wrong (largest |difference| in log-probabilities against the reference)")
for label, kw in [("rotary pairs interleaved", dict(interleaved=True)), ("rotary theta 10,000 instead of 100,000", dict(th=10000.0)),
                  ("key-value heads tiled, not repeated in place", dict(kv_tile=True)), ("layer norm instead of RMSNorm", dict(layer_norm=True))]:
    with torch.no_grad():
        Sx = forward(ids, **kw)
    print(f"  {label:<46} {float((torch.log_softmax(Sx, -1) - lr).abs().max()):10.3e}  same top token everywhere: {bool((Sx.argmax(-1) == R.argmax(-1)).all())}")

from _pool import corpora
c = corpora(); t = tok(); g = tk.encode(t.decode([int(i) for i in c["held_b"]]))
print(f"\nPart D: held-out Carroll, {len(g):,} SmolLM2 tokens ({len(g) / 10919:.3f} per character)")
for ctx in (98, 1024):              # 98 SmolLM2 tokens is about the characters of Tiny's 129 (0.279 against 0.366 per character)
    tot, chars, nn_ = 0.0, 0, 0
    for a in range(0, len(g) - 1, ctx):
        w = g[a:a + ctx]
        if len(w) < 2: break
        with torch.no_grad():
            lp = torch.log_softmax(forward(w).double(), -1)
        s_ = -lp[torch.arange(len(w) - 1), torch.tensor(w[1:])]
        tot += float(s_.sum()); nn_ += len(w) - 1; chars += len(tk.decode(w[1:]))
    print(f"  windows of {ctx:4d}: mean surprise {tot / nn_:.4f} nats per token; {tot / math.log(2) / chars:.4f} bits per character", flush=True)
