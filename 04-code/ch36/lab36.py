"""Lab 36: find GPT-2 small's induction heads, make SmolLM2-135M learn a task from four examples, break it by giving every
example a different separator, and check that an agreeing note is taken.  CPU, one thread, 32-bit, seed 1865.
The lab makes no API call; it reads the key file only to check that the key appears in no file under 04-code/ch36."""
from _common import *
import torch, numpy as np
torch.set_num_threads(1); torch.manual_seed(SEED)
from transformers import AutoTokenizer, AutoModelForCausalLM
from fewshot_lib import PAIRS, build, full_text
ok = 0; N = 0
def check(cond, msg):
    global ok, N
    N += 1; ok += bool(cond); print(("[pass] " if cond else "[FAIL] ") + msg, flush=True)
# 1 induction heads in GPT-2 small: 5 draws of 50 random tokens shown twice
tk = AutoTokenizer.from_pretrained(CKPT("gpt2"))
m = AutoModelForCausalLM.from_pretrained(CKPT("gpt2"), dtype=torch.float32, attn_implementation="eager").eval()
rng = np.random.default_rng(SEED); n = 50; ind = 0; loss2 = []
for _ in range(5):
    run = rng.integers(1000, 30000, size=n).tolist(); ids = torch.tensor([run + run])
    with torch.no_grad(): o = m(ids, output_attentions=True)
    A = torch.stack([a[0] for a in o.attentions]); q = torch.arange(n, 2 * n)
    ind = ind + A[:, :, q, q - n + 1].mean(-1) / 5
    lp = torch.log_softmax(o.logits[0].double(), -1); loss2.append(float(-lp[q[:-1], ids[0, q[1:]]].mean()))
best = float(ind.max()); l, h = divmod(int(ind.argmax()), ind.shape[1])
print(f"  best induction head {l}.{h}: {best:.3f}; loss on the repeat {np.mean(loss2):.3f} nats", flush=True)
check(best > 0.8 and np.mean(loss2) < 1.0, "GPT-2 small has an induction head above 0.8 and predicts a random repeat at under 1 nat a token")
# 2-3 few-shot and its breakage, SmolLM2-135M
tk = AutoTokenizer.from_pretrained(CKPT("smollm2")); m = AutoModelForCausalLM.from_pretrained(CKPT("smollm2"), dtype=torch.float32).eval()
def acc(fmt, k=4):
    r = np.random.default_rng(SEED); c = 0
    for qi in range(len(PAIRS)):
        text = full_text(build(qi, k, r, fmt), fmt, k)
        with torch.no_grad(): t = int(m(torch.tensor([tk.encode(text)])).logits[0, -1].argmax())
        c += t == tk.encode(" " + PAIRS[qi][1])[0]
    return c
a0, a4, am = acc("arrow", 0), acc("arrow"), acc("mixed")
print(f"  SmolLM2 antonyms right of 24: no examples {a0}, four arrow examples {a4}, four with mixed separators {am}", flush=True)
check(a0 == 0 and a4 >= 18, "SmolLM2-135M goes from 0 to at least 18 of 24 with four examples")
check(a4 - am >= 6, "giving every example a different separator costs at least 6 of 24 (the breakage)")
# 4 an agreeing note
F = [("France","Paris"),("Japan","Tokyo"),("Peru","Lima"),("Kenya","Nairobi")]; c0 = c1 = 0
for c, city in F:
    for note, ctr in (("", 0), (f"Note: the capital of {c} is {city}.\n\n", 1)):
        with torch.no_grad(): t = int(m(torch.tensor([tk.encode(note + f"The capital of {c} is")])).logits[0, -1].argmax())
        if t == tk.encode(" " + city)[0]: c0 += ctr == 0; c1 += ctr == 1
print(f"  capitals right of 4: closed book {c0}, with an agreeing note {c1}", flush=True)
check(c1 == 4, "an agreeing note is taken 4 of 4")
# 5 the key
key = None
for line in open(os.path.join(HERE, "..", "..", ".env")):
    if line.startswith("ANTHROPIC_API_KEY="): key = line.split("=", 1)[1].strip().strip('"')
hit = False
for root, _, files in os.walk(HERE):
    for f in files:
        try: hit |= key in open(os.path.join(root, f), errors="ignore").read()
        except Exception: pass
check(key and not hit, "the API key was read from .env and its string appears in no file under 04-code/ch36")
print(f"\n{ok} of {N} checkpoints pass")
