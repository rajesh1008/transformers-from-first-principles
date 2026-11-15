"""Lab 33: predict, break, check.  Run from 04-code/ch33 after quant_gpt2.py (reads outputs/quant_gpt2.json); about two
minutes.  Part 1: predict a model's size.  Part 2: two breakages (the token table rounded for the output scores; GPTQ
calibrated on random token ids).  Part 3: checkpoints.  No API call; the key is read from .env only to check its absence
from every file under 04-code/ch33."""
from _common import *
import torch, glob, math
torch.set_num_threads(1)
from tiny.load import GPT2
from tiny import quant as Q
from transformers import AutoTokenizer
ok = []
def check(cond, text):
    ok.append(bool(cond)); print(("[pass] " if cond else "[FAIL] ") + text)
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2")); g = held_text_ids(tk)
R = load_json("quant_gpt2"); R.update(load_json("quant_gptq"))
print("Part 1. predict, then read")
w = gm.W["h.0.mlp.c_fc.weight"]
for bits, gr in ((8, None), (4, 128), (4, 32), (3, 32)):
    b = Q.storage_bits(tuple(w.shape), bits, gr)
    print(f"  768 x 3,072 map at {bits} bits, {'per output channel' if gr is None else f'per {gr} inputs'}: {b / 8 / 1e6:.3f} MB, "
          f"{b / w.numel():.3f} bits per weight")
print("\nPart 2. breakages")
win = g[:1024]
def bpc1(W):
    with torch.no_grad():
        lp = torch.log_softmax(Q.forward_q(gm, win, W).double(), -1)
    return float(-lp[torch.arange(len(win) - 1), torch.tensor(win[1:])].sum()) / math.log(2) / len(tk.decode(win[1:]))
base = bpc1(None)
E4 = Q.quantize(gm.W["wte.weight"].t().contiguous(), 4, 128).t().contiguous()
b4 = bpc1({"wte.weight": E4})
print(f"  first 1,024-token window: 32-bit {base:.4f}; token table at 4 bits per 128 (maps in 32 bits) {b4:.4f} ({b4 - base:+.4f})")
F = R["F"]
print(f"  3-bit per 128, GPTQ calibrated on the Carroll pool {F['3 128']:.4f}; on random token ids {F['3 128 cal random token ids']:.4f} "
      f"(round-to-nearest {R['B']['3 128']:.4f})")
print("\nPart 3. checkpoints")
q = Q.quantize(w, 8, None); step = w.abs().amax(0) / 127
check(bool(((q - w).abs() <= step / 2 + 1e-7).all()), "8-bit per output channel: every weight within half a step of its original")
check(abs(Q.storage_bits((768, 3072), 4, 32) / (768 * 3072) - 4.5) < 1e-9, "4 bits per 32 inputs costs 4.5 bits per weight with 16-bit scales")
check(R["B"]["4 tensor"] > R["B"]["4 None"] > R["B"]["4 128"] > R["B"]["4 32"], "at 4 bits, smaller groups gave lower bits per character (table > channel > 128 > 32)")
check(b4 - base > 1.0, f"rounding the token table to 4 bits breaks GPT-2 (+{b4 - base:.2f} bits per character on one window)")
check(F["3 128"] < R["B"]["3 128"], "GPTQ beat round-to-nearest at 3 bits per 128 inputs")
key = [l.split("=", 1)[1].strip().strip('"') for l in open(os.path.join(HERE, "..", "..", ".env")) if l.startswith("ANTHROPIC_API_KEY")][0]
clean = all(key not in open(f, errors="ignore").read() for f in glob.glob(os.path.join(HERE, "**", "*"), recursive=True) if os.path.isfile(f))
check(clean, "the API key was read from .env and its string appears in no file under 04-code/ch33")
print(f"\n{sum(ok)} of {len(ok)} checkpoints pass")
assert all(ok)
