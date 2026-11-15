"""Chapter 35, 'What grows when the window grows': time one forward pass of Tiny (lengths 128 to 1,024, the size of its
position table) and of SmolLM2-135M (128 to 4,096, through longfwd.py's blocked attention) and count what grows: attention
scores per layer, cache bytes, and the multiply-adds in the linear maps against those in the score tables.
CPU, one thread, 32-bit, seed 1865.  Times are the best of `reps` runs on a machine shared with other processes."""
from _common import *
import time, torch, json
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
from tiny.load import Llama
from transformers import AutoTokenizer
from longfwd import token_losses
from _pool import corpora

def best(f, reps):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); t.append(time.perf_counter() - t0)
    return min(t)

R = {"tiny": {}, "smol": {}}
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
g = [int(i) for i in corpora()["glass"]]
print("Tiny (12 layers, width 128, 4 heads), one forward pass over n tokens, best of 3:")
for n in [128, 256, 512, 1024]:
    ids = g[:n]
    with torch.no_grad(): s = best(lambda: tt.scores(ids), 3)
    R["tiny"][n] = s; print(f"  n {n:5d}  {s*1000:9.1f} ms   {s/n*1e6:8.1f} us per token", flush=True)

tk = AutoTokenizer.from_pretrained(CKPT("smollm2")); m = Llama.from_dir(CKPT("smollm2"))
for k in list(m.W): m.W[k] = m.W[k].float()
m.emb = m.W["model.embed_tokens.weight"]; m.out = m.W.get("lm_head.weight", m.emb)
ids = tk.encode(glass_text())
print("SmolLM2-135M (30 layers, width 576, 9 query heads, 3 key/value heads), loss of every token in one pass, blocked attention:")
for n in [128, 256, 512, 1024, 2048, 4096]:
    reps = 3 if n <= 1024 else 1
    with torch.no_grad(): s = best(lambda: token_losses(m, ids[:n]), reps)
    R["smol"][n] = s; print(f"  n {n:5d}  {s:8.2f} s   {s/n*1e3:7.2f} ms per token", flush=True)

# what grows, by arithmetic (SmolLM2-135M config: 30 layers, d 576, 9 heads of 64, 3 kv heads, ffn 1,536, vocab 49,152)
L, d, H, KV, hd, F, V = 30, 576, 9, 3, 64, 1536, 49152
lin = L * (d * H * hd + 2 * d * KV * hd + H * hd * d + 3 * d * F)     # multiply-adds per token in the linear maps
print("SmolLM2-135M by arithmetic: per token, linear maps", f"{lin:,}", "multiply-adds (plus", f"{d*V:,}", "for the output table)")
print(" n          score entries per layer   attention multiply-adds per token (QK and AV)   share of the layer maps   cache bytes (bf16)")
A = {}
for n in [1024, 4096, 8192, 32768, 131072, 1048576]:
    att = L * H * 2 * hd * n / 2            # average token attends to n/2 earlier tokens, QK^T and AV
    cache = 2 * L * KV * hd * 2 * n
    A[n] = dict(scores=H * n * n, att=att, ratio=att / lin, cache=cache)
    print(f" {n:9,d}  {H*n*n:24,d}   {att:28,.0f}   {att/lin:10.2f}   {cache:16,d}")
R["arith"] = {str(k): v for k, v in A.items()}; R["lin"] = lin
save_json("cost_curve", R)
