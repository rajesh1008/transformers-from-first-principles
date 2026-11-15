"""Chapter 35, 35.3: SmolLM2-135M (rotary positions, base 100,000, max_position_embeddings 8,192 in its config) on one long
stretch of Through the Looking-Glass, scored at 16,384 tokens: plain rotary angles, positions divided by 2 (position
interpolation) and by 4, and a raised base (theta x 4 and x 16).  Mean loss per position band.  CPU, one thread, 32-bit.
  python3 smol_long.py MODE      MODE = plain | pi2 | pi4 | ntk4 | ntk16"""
from _common import *
import sys, time, torch
torch.set_num_threads(1)
from tiny.load import Llama
from transformers import AutoTokenizer
from longfwd import token_losses
MODE = sys.argv[1]; N = int(os.environ.get("N", "16384"))
tk = AutoTokenizer.from_pretrained(CKPT("smollm2")); m = Llama.from_dir(CKPT("smollm2"))
for k in list(m.W): m.W[k] = m.W[k].float()
m.emb = m.W["model.embed_tokens.weight"]; m.out = m.W.get("lm_head.weight", m.emb)
ids = tk.encode(glass_text())
print(f"Looking-Glass in SmolLM2's tokenizer: {len(ids):,} tokens; scoring the first {N:,}", flush=True)
ids = ids[:N]
kw = dict(plain={}, pi2=dict(scale=2.0), pi4=dict(scale=4.0), ntk4=dict(theta=m.theta * 4), ntk16=dict(theta=m.theta * 16))[MODE]
t0 = time.perf_counter(); L = token_losses(m, ids, **kw); dt = time.perf_counter() - t0
B = [(0, 511), (512, 2047), (2048, 4095), (4096, 8191), (8192, 12287), (12288, N - 2)]
R = {"mode": MODE, "N": N, "seconds": dt}
for a, b in B:
    if a >= len(L): break
    v = float(L[a:b + 1].mean()); R[f"{a}-{b}"] = v; print(f"  {MODE:6s} positions {a:5d} to {b:5d}: {v:.4f} nats", flush=True)
torch.save(L, os.path.join(HERE, "outputs", f"smol_long_{MODE}_{N}.pt"))
save_json(f"smol_long_{MODE}_{N}", R); print(f"{MODE}: {dt:.0f} s")
