"""Chapter 33, gptq_check.py: does GPTQ do its own job?  For four of GPT-2's maps, the output error on the calibration
rows, trace((W - Q)^T H (W - Q)), and the weights' own squared error, for round-to-nearest and GPTQ at 4 and 3 bits per 128
inputs.  Same calibration as quant_gptq.py (16 x 256 tokens of the Carroll pool).  CPU, one thread, seed 1865, 11 Nov 2026."""
from _common import *
import torch
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.load import GPT2
from tiny import quant as Q
from transformers import AutoTokenizer
from _pool import corpora
from tiny.tokenizer import load_bpe
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2"))
tt = load_bpe(TINY_BPE); C = corpora()
pool_ids = tk.encode(tt.decode([int(i) for i in C["pool"][100000:112000]]))
cal = [pool_ids[i * 256:(i + 1) * 256] for i in range(16)]
XtX, amax = Q.collect_inputs(gm, cal)
for name in ("h.2.mlp.c_fc", "h.2.attn.c_attn", "h.11.mlp.c_proj", "h.0.attn.c_proj"):
    w = gm.W[name + ".weight"]; H = XtX[name]
    for bits, gr in ((4, 128), (3, 128)):
        r = Q.quantize(w, bits, gr); g = Q.gptq(w, H, bits, gr)
        e = lambda q: float(torch.trace((w - q).double().t() @ H @ (w - q).double()))
        print(name, bits, gr, f"calib output error RTN {e(r):.4g} GPTQ {e(g):.4g}; weight err RTN {float(((w-r)**2).sum()):.4g} GPTQ {float(((w-g)**2).sum()):.4g}", flush=True)
