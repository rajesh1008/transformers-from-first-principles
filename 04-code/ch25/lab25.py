"""Lab 25: five ways to fool yourself with an evaluation, and the checkpoint.  Reads the outputs of evaluate.py,
compare.py, contaminate.py and bench.py (run them first; RUN.txt), recomputes the numbers that need the model from the
saved tables, and stops with an error if any checkpoint fails.  32-bit, one thread.  About thirty seconds."""
import _common
import os, math, json, glob
import numpy as np, torch
torch.set_num_threads(1)
from tiny import eval as E
from _pool import corpora, heldout_sets
from _common import HERE, sentence_ids, load_json
EV, CP, CT, BN = (load_json(n) for n in ("evaluate", "compare", "contaminate", "bench"))
c = corpora()
tt = E.load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
print("Part 1. predict, then read")
L = EV["texts"]["carroll"]["2500"]["loss"]
print(f"  perplexity of a loss of {L:.4f} nats: e^{L:.4f} = {math.exp(L):.1f}")
print(f"  the same loss in bits per token: {L / math.log(2):.4f}; at 0.367 tokens per character, bits per character {L / math.log(2) * 0.367:.4f}")
sdw = EV["boot_glass"]["window_sd"]
print(f"  windows of Looking-Glass for a standard error of 0.01, with a window sd of {sdw:.4f}: {math.ceil((sdw / 0.01) ** 2)}")

print("\nPart 2. breakages")
t = EV["texts"]
print(f"  Breakage 1 (scoring on text the model read): pool {t['pool']['2500']['loss']:.4f}, Carroll held-out {t['carroll']['2500']['loss']:.4f}, Looking-Glass {t['glass']['2500']['loss']:.4f}")
b = EV["boot_carroll"]
print(f"  Breakage 2 (predictions treated as independent): standard error {b['naive']['se']:.4f} (naive) against {b['window']['se']:.4f} (whole windows); "
      f"95% {b['naive']['low']:.4f} to {b['naive']['high']:.4f} against {b['window']['low']:.4f} to {b['window']['high']:.4f}")
pr = CP["pairs"]["wd0.1 o11 2,500 | o11 2,500 | glass"]; d = CP["wd_orders_2,500_glass"]["diffs"]
print(f"  Breakage 3 (a paired test read as the whole story): AdamW 0.1 minus Adam, order 11, Looking-Glass {pr['diff']:+.4f}, paired 95% {pr['paired'][0]:+.4f} to {pr['paired'][1]:+.4f}, "
      f"sign test p = {pr['p_sign']:.2g}; the same difference under orders 1865, 7, 11: {', '.join(f'{x:+.4f}' for x in d)}")
tr = EV["temperature"]["results"]
print(f"  Breakage 4 (a temperature fitted on one text): T {EV['temperature']['best']} lowers the loss on held-out Looking-Glass {tr['glass second half']['loss1']:.4f} -> {tr['glass second half']['lossT']:.4f}, "
      f"raises it on the Carroll held-out set {tr['carroll']['loss1']:.4f} -> {tr['carroll']['lossT']:.4f} and triples the pool's ECE {tr['pool']['ece1']:.4f} -> {tr['pool']['eceT']:.4f}")
u = BN["results"]["unigram"]
print(f"  Breakage 5 (a benchmark whose right answers are shorter): unigram counts score {u['glass_sum']:.3f} on the cloze items with summed log-probabilities and {u['glass_mean']:.3f} "
      f"with the average per token (chance 0.250); right answers average {BN['tokens_true']:.2f} tokens, distractors {BN['tokens_distr']:.2f}")

print("\nPart 3. checkpoints")
ok = []
def check(cond, msg):
    ok.append(bool(cond)); print(("[pass] " if cond else "[FAIL] ") + msg)
G16, HB16 = heldout_sets()
with torch.no_grad():
    l16 = float(tt.loss(torch.tensor(HB16)))
check(abs(l16 - 3.7296) < 5e-5, f"the saved 2,500-step tables reproduce Chapter 24's held-out loss on its 16 windows: {l16:.4f}")
s = sentence_ids()
with torch.no_grad():
    lp = torch.log_softmax(tt.scores(torch.tensor(s[:-1])).double(), -1)
p = np.array([float(lp[i, s[i + 1]].exp()) for i in range(len(s) - 1)])
ppl, gm = math.exp(-np.log(p).mean()), math.exp(np.log(p).mean())
check(abs(ppl * gm - 1) < 1e-9, f"perplexity is one over the geometric mean of the probabilities (running sentence): {ppl:.1f} x {gm:.5f} = {ppl * gm:.6f}")
bl = [r for r in EV["bands"]["carroll"] if r.get("n")]
bigram = sum(r["bigram"] * r["n"] for r in bl) / sum(r["n"] for r in bl)
check(math.exp(L) < math.exp(bigram), f"held-out perplexity on all 31 Carroll windows beats the bigram's: {math.exp(L):.1f} < {math.exp(bigram):.1f}")
fl = CT["ngram_13_pool + exact copies"]["flagged"]
check(fl == list(range(0, 31, 4)), f"the 13-gram check flags exactly the 8 planted windows when exact copies are planted: {fl}")
check(CT["ngram_13_pool + altered copies"]["flagged"] == [] and CT["altered x4"]["planted"] < CT["control"]["planted"] - 0.05,
      f"altered copies escape the check yet lower the planted windows' loss: {CT['control']['planted']:.4f} -> {CT['altered x4']['planted']:.4f}")
q = E.score(tt, E.windows(c["held_b"]))
b1, b2 = E.bootstrap(q["surprise"], q["window"]), E.bootstrap(q["surprise"], q["window"])
check(b1 == b2 and abs(b1["mean"] - L) < 1e-9, f"the window bootstrap repeats exactly with seed 1865: {b1['low']:.4f} to {b1['high']:.4f}")
keyfile = os.path.join(HERE, "..", "..", ".env")
key = [l.split("=", 1)[1].strip() for l in open(keyfile) if l.startswith("ANTHROPIC_API_KEY=")][0] if os.path.exists(keyfile) else None
leak = [f for f in glob.glob(os.path.join(HERE, "**", "*"), recursive=True) if os.path.isfile(f) and not f.endswith(".pt") and key and key in open(f, errors="ignore").read()]
check(key is None or not leak, "the API key string appears in no file under 04-code/ch25")
print(f"\n{sum(ok)} of {len(ok)} checkpoints pass")
assert all(ok)
