"""Chapter 29, sections 29.1 to 29.4: loading GPT-2 small (openai-community/gpt2) from its released files and running it
with the book's own forward pass, written from the tables, then checking it against a reference implementation
(Hugging Face transformers' GPT2LMHeadModel) on the running sentence.  32-bit, one thread.  The files were fetched with
curl into ckpt/gpt2/ (not shipped in the code zip; RUN.txt gives the commands)."""
from _common import *
import torch, json, struct, hashlib, math, time
torch.set_num_threads(1)
D = os.path.join(HERE, "ckpt", "gpt2")
out = {}

print("Part A: the file")
raw = open(os.path.join(D, "model.safetensors"), "rb").read()
n = struct.unpack("<Q", raw[:8])[0]
head = json.loads(raw[8:8 + n]); meta = head.pop("__metadata__", None)
print(f"  {len(raw):,} bytes; header {n:,} bytes; {len(head)} tables; metadata {meta}; sha256 {hashlib.sha256(raw).hexdigest()[:16]}")
dtypes = sorted({v['dtype'] for v in head.values()})
total = sum(int(np.prod(v["shape"])) for v in head.values())
print(f"  number formats {dtypes}; numbers stored {total:,}; data bytes {max(v['data_offsets'][1] for v in head.values()):,}")
for k in ["wte.weight", "wpe.weight", "ln_f.weight"] + [k for k in sorted(head) if k.startswith("h.0.")]:
    print(f"    {k:<24} {head[k]['dtype']} {head[k]['shape']}")
out["file"] = dict(bytes=len(raw), header=n, tables=len(head), numbers=total, dtypes=dtypes)

from safetensors.numpy import load_file
W = load_file(os.path.join(D, "model.safetensors"))
W = {k: torch.from_numpy(v.astype(np.float32)) for k, v in W.items()}
cfg = json.load(open(os.path.join(D, "config.json")))
L, H, d, eps = cfg["n_layer"], cfg["n_head"], cfg["n_embd"], cfg["layer_norm_epsilon"]
masks = [k for k in W if k.endswith("attn.bias") and W[k].dim() == 4]
print(f"  tables that are not learned numbers (causal-mask buffers): {len(masks)}")

def ln(x, p, eps=eps):
    m = x.mean(-1, keepdim=True); v = ((x - m) ** 2).mean(-1, keepdim=True)
    return (x - m) / torch.sqrt(v + eps) * W[p + ".weight"] + W[p + ".bias"]

def gelu_tanh(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

def forward(ids, transpose=False, gelu=gelu_tanh, eps_=eps, pos_offset=0, skip_ln_f=False):
    """The book's forward pass for GPT-2 small, from the tables by name.  GPT-2 stores every linear map as
    [inputs, outputs] (its 'Conv1D'), so x @ W is right as stored; transpose=True is the breakage."""
    T = len(ids); idx = torch.tensor(ids)
    x = W["wte.weight"][idx] + W["wpe.weight"][pos_offset:pos_offset + T]
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    mm = (lambda a, w: a @ w.T) if transpose else (lambda a, w: a @ w)
    for l in range(L):
        p = f"h.{l}."
        a = ln(x, p + "ln_1", eps_)
        qkv = mm(a, W[p + "attn.c_attn.weight"]) + W[p + "attn.c_attn.bias"]
        q, k, v = qkv.split(d, -1)
        sh = lambda z: z.view(T, H, d // H).transpose(0, 1)
        q, k, v = sh(q), sh(k), sh(v)
        s = q @ k.transpose(-1, -2) / math.sqrt(d // H)
        s = s.masked_fill(~mask, float("-inf"))
        o = (torch.softmax(s, -1) @ v).transpose(0, 1).reshape(T, d)
        x = x + mm(o, W[p + "attn.c_proj.weight"]) + W[p + "attn.c_proj.bias"]
        f = gelu(mm(ln(x, p + "ln_2", eps_), W[p + "mlp.c_fc.weight"]) + W[p + "mlp.c_fc.bias"])
        x = x + mm(f, W[p + "mlp.c_proj.weight"]) + W[p + "mlp.c_proj.bias"]
    if not skip_ln_f:
        x = ln(x, "ln_f", eps_)
    return x @ W["wte.weight"].T                       # tied output table

from transformers import AutoTokenizer, GPT2LMHeadModel
tk = AutoTokenizer.from_pretrained(D)
ids = tk.encode(SENTENCE)
print(f"\nPart B: the running sentence {SENTENCE!r} in GPT-2's tokenizer: {ids} -> {[tk.decode([i]) for i in ids]}")
t0 = time.perf_counter(); with_ = forward(ids); t_own = time.perf_counter() - t0
ref = GPT2LMHeadModel.from_pretrained(D, torch_dtype=torch.float32).eval()
with torch.no_grad():
    R = ref(torch.tensor([ids])).logits[0]
diff = (with_ - R).abs()
print(f"  own forward vs reference: largest |difference| in scores {diff.max():.3e}; mean {diff.mean():.3e}; same top token at every "
      f"position: {bool((with_.argmax(-1) == R.argmax(-1)).all())}; own pass {t_own * 1000:.0f} ms")
lp, lr = torch.log_softmax(with_, -1), torch.log_softmax(R, -1)
print(f"  largest |difference| in log-probabilities {(lp - lr).abs().max():.3e}")
out["check"] = dict(max_abs=float(diff.max()), mean_abs=float(diff.mean()), max_logp=float((lp - lr).abs().max()))
pre = tk.encode("the cat sat on the")
p = torch.softmax(forward(pre)[-1].double(), -1)
Hn = float(-(p * torch.log(p.clamp_min(1e-300))).sum())
top = torch.argsort(-p)[:10]
print(f"\n  after 'the cat sat on the' ({len(pre)} tokens): entropy {Hn:.3f} nats (about {math.exp(Hn):.0f} equally likely choices)")
print("  top 10: " + ", ".join(f"{tk.decode([int(j)])!r} {float(p[j]):.4f}" for j in top))
mat = tk.encode(" mat")[0]; rk = int((p > p[mat]).sum()) + 1
print(f"  ' mat' share {float(p[mat]):.4f}, rank {rk}")
out["next"] = dict(entropy=Hn, top=[(tk.decode([int(j)]), float(p[j])) for j in top], mat=float(p[mat]), mat_rank=rk)

print("\nPart C: five ways to load it wrong (largest |difference| in log-probabilities against the reference, and top token kept?)")
def cmp(label, S):
    l = torch.log_softmax(S, -1); dd = float((l - lr).abs().max()); same = bool((S.argmax(-1) == R.argmax(-1)).all())
    print(f"  {label:<46} {dd:10.3e}  same top token everywhere: {same}"); out["break " + label] = dict(max_logp=dd, same=same)
try:
    cmp("linear maps transposed", forward(ids, transpose=True))
except RuntimeError as e:
    print(f"  linear maps transposed                         error: {str(e)[:90]}")
def tr_sq(ids):
    """only the square maps transposed (c_proj of attention: 768 x 768), which raises no shape error"""
    keep = {k: W[k] for k in W if k.endswith("attn.c_proj.weight")}
    for k in keep: W[k] = keep[k].T.contiguous()
    try: return forward(ids)
    finally:
        for k in keep: W[k] = keep[k]
cmp("square attention output maps transposed", tr_sq(ids))
cmp("exact GELU (erf) instead of tanh form", forward(ids, gelu=torch.nn.functional.gelu))
cmp("layer-norm eps 1e-6 instead of 1e-5", forward(ids, eps_=1e-6))
cmp("positions offset by one", forward(ids, pos_offset=1))
cmp("final layer norm skipped", forward(ids, skip_ln_f=True))
t = tok(); tiny_ids = t.encode(SENTENCE)
print(f"  Tiny's tokenizer on the same sentence gives {tiny_ids}: as GPT-2 ids these read {tk.decode(tiny_ids)!r}")
save_json("load_gpt2", out)
