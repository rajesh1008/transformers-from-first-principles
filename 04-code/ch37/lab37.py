"""Lab 37: measure Tiny's calibration and repair it with one temperature; see that Tiny's confidence barely knows which text
it has read; force SmolLM2-135M to answer at temperature 0 and watch a wrong answer come out more confident than right ones.
CPU, one thread, 32-bit, seed 1865.  The lab makes no API call; it reads the key file only to check that the key appears
in no file under 04-code/ch37."""
from _common import *
import torch, numpy as np
torch.set_num_threads(1); torch.manual_seed(SEED)
from tiny.eval import load
from _pool import corpora
from transformers import AutoTokenizer, AutoModelForCausalLM
ok = 0; N = 0
def check(cond, msg):
    global ok, N
    N += 1; ok += bool(cond); print(("[pass] " if cond else "[FAIL] ") + msg, flush=True)
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")); c = corpora()
def stats(ids, n, temp=1.0, start=0):
    conf = []; acc = []; loss = []
    for w in range(start, start + n):
        ch = ids[w * 128: w * 128 + 129]
        with torch.no_grad(): lp = torch.log_softmax(tt.scores(ch)[:-1].double() / temp, -1)
        tg = torch.tensor(ch[1:]); cf, top = lp.max(-1)
        conf.append(cf.exp()); acc.append((top == tg).double()); loss.append(-lp[torch.arange(128), tg])
    return torch.cat(conf), torch.cat(acc), float(torch.cat(loss).mean())
def ece(cf, ac):
    e = 0.0
    for i in range(10):
        m = (cf > i / 10) & (cf <= (i + 1) / 10)
        if m.any(): e += m.double().mean().item() * abs(float(ac[m].mean()) - float(cf[m].mean()))
    return e
g = [int(i) for i in c["glass"]]
cf, ac, _ = stats(g, 40)
e1 = ece(cf, ac); print(f"  Tiny on 40 Looking-Glass windows: accuracy {float(ac.mean()):.4f}, confidence {float(cf.mean()):.4f}, ECE {e1:.4f}", flush=True)
check(float(cf.mean()) > float(ac.mean()) and e1 < 0.06, "Tiny is overconfident on unseen text, by a few points")
best = min(((t, stats(g, 40, t)[2]) for t in (0.9, 1.0, 1.1, 1.2, 1.3)), key=lambda x: x[1])[0]
cf2, ac2, _ = stats(g, 40, best, start=40); cf1, ac1, _ = stats(g, 40, 1.0, start=40)
print(f"  temperature chosen on windows 0-39: {best}; ECE on windows 40-79: {ece(cf1, ac1):.4f} -> {ece(cf2, ac2):.4f}", flush=True)
check(ece(cf2, ac2) < ece(cf1, ac1) and torch.equal(ac1, ac2), "one temperature lowers the calibration error and leaves every top choice unchanged")
p = [int(i) for i in c["pool"]]
cfp, acp, _ = stats(p, 40)
print(f"  trained-on text: accuracy {float(acp.mean()):.4f}, confidence {float(cfp.mean()):.4f}; unseen: {float(ac.mean()):.4f}, {float(cf.mean()):.4f}", flush=True)
check((float(acp.mean()) - float(ac.mean())) > 2 * (float(cfp.mean()) - float(cf.mean())), "accuracy falls more than twice as much as confidence when Tiny leaves its training text")
tk = AutoTokenizer.from_pretrained(CKPT("smollm2")); m = AutoModelForCausalLM.from_pretrained(CKPT("smollm2"), dtype=torch.float32).eval()
def gen(cn):
    ids = torch.tensor([tk.encode(f"The capital of {cn} is")])
    with torch.no_grad(): o = m.generate(ids, max_new_tokens=10, do_sample=False, output_scores=True, return_dict_in_generate=True, pad_token_id=tk.eos_token_id)
    new = o.sequences[0, ids.shape[1]:]
    return tk.decode(new).strip().replace("\n", " "), float(np.mean([float(torch.softmax(s[0].double(), -1)[t]) for s, t in zip(o.scores, new)]))
mol, vm = gen("Moldova"); lao, vl = gen("Laos")
print(f"  Moldova: {mol!r} ({vm:.3f})   Laos: {lao!r} ({vl:.3f})", flush=True)
check("chis" not in mol.lower() and "vientiane" in lao.lower() and vm > vl, "at temperature 0 SmolLM2 names the wrong city for Moldova, more confidently than the right one for Laos")
key = None
for line in open(os.path.join(HERE, "..", "..", ".env")):
    if line.startswith("ANTHROPIC_API_KEY="): key = line.split("=", 1)[1].strip().strip('"')
hit = False
for root, _, files in os.walk(HERE):
    for f in files:
        try: hit |= key in open(os.path.join(root, f), errors="ignore").read()
        except Exception: pass
check(key and not hit, "the API key was read from .env and its string appears in no file under 04-code/ch37")
print(f"\n{ok} of {N} checkpoints pass")
