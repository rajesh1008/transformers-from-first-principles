"""Lab 32: predict, break, check.  Run from 04-code/ch32 (about a minute).
Part 1: predict tokens per pass from an acceptance rate.  Part 2: two breakages (the draft that always guesses ' the';
a cache not cut back after a rejection).  Part 3: checkpoints.  No API call; the key is read from .env only to check
its absence from every file under 04-code/ch32."""
from _common import *
import torch, glob, numpy as np
torch.set_num_threads(1)
from tiny.load import GPT2
from tiny.cache import greedy, new_cache
from tiny import speculate as SP
from spec_greedy_lib import fwd_layers
from transformers import AutoTokenizer
ok = []
def check(cond, text):
    ok.append(bool(cond)); print(("[pass] " if cond else "[FAIL] ") + text)
print("Part 1. predict, then read")
for a in (0.3, 0.6, 0.8):
    print(f"  acceptance {a}, k = 4: expected tokens per pass {(1 - a ** 5) / (1 - a):.2f}")
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2")); g = held_text_ids(tk)
THE = tk.encode(" the")[0]
prompts = [tk.encode(SENTENCE), g[800:864]]
refs = [greedy(gm, p, 32, True)[0] for p in prompts]
print("\nPart 2. breakages")
the = [SP.speculate(gm, p, 32, lambda ids, k: ([THE] * k, None), 4) for p in prompts]
for (ids, st), ref in zip(the, refs):
    print(f"  always ' the': same as greedy {ids == ref}; accepted {st['accepted']}/{st['proposed']}; {32 / (st['passes'] - 1):.2f} tokens per pass")
def stale(model, prompt, n, k=4):
    """speculate() with step 6 left out: rejected guesses' keys and values stay in the cache."""
    ids = list(prompt); cache = new_cache(model); pos = 0
    with torch.no_grad():
        s = model.forward(ids, cache, 0)[-1]; ids.append(int(s.argmax())); pos = len(ids) - 1
        while len(ids) - len(prompt) < n:
            guess = SP.lookup_draft(ids, k, THE)
            rows = model.forward([ids[-1]] + guess, cache, pos); pos += k + 1
            m, t = SP.accept_greedy(guess, rows); ids += guess[:m] + [t]
    return ids[:len(prompt) + n]
st_out = [stale(gm, p, 32) for p in prompts]
for p, o, ref in zip(prompts, st_out, refs):
    j = next((i for i in range(len(ref)) if o[i] != ref[i]), None)
    print(f"  cache not cut back: same as greedy {o == ref}" + (f"; first differs at new token {j - len(p) + 1}: "
          f"{tk.decode(o[len(p):j + 1])!r} against {tk.decode(ref[len(p):j + 1])!r}" if j is not None else ""))
print("\nPart 3. checkpoints")
look = [SP.speculate(gm, p, 32, lambda ids, k: (SP.lookup_draft(ids, k, THE), None), 4)[0] for p in prompts]
check(all(a == b for a, b in zip(look, refs)), "speculative greedy with prompt lookup writes exactly plain greedy's tokens on two prompts")
check(all(a[0] == b for a, b in zip(the, refs)) and all(s["accepted"] / s["proposed"] < 0.1 for _, s in the),
      "the ' the' draft keeps the tokens and loses the speed (acceptance below 0.1)")
check(any(o != ref for o, ref in zip(st_out, refs)), "a cache not cut back after a rejection changes the output")
with torch.no_grad():
    p = SP.probs(gm.forward(prompts[0])[-1], 1.0); q = SP.probs(fwd_layers(gm, prompts[0], [None] * 6, 0, 6)[-1], 1.0)
a = np.minimum(p, q).sum(); r = np.maximum(p - q, 0); r /= r.sum()
d = np.abs(np.minimum(p, q) + (1 - a) * r - p).max()
check(d < 1e-12, f"the speculative-sampling rule's output equals the target's shares exactly after the running sentence (max diff {d:.1e})")
b = q * np.minimum(1, p / q) + (1 - a) * p
with torch.no_grad():
    p2 = SP.probs(gm.forward(g[800:864])[-1], 1.0); q2 = SP.probs(fwd_layers(gm, g[800:864], [None] * 6, 0, 6)[-1], 1.0)
a2 = np.minimum(p2, q2).sum(); b2 = q2 * np.minimum(1, p2 / q2) + (1 - a2) * p2
tv1, tv2 = 0.5 * np.abs(b - p).sum(), 0.5 * np.abs(b2 - p2).sum()
check(min(tv1, tv2) > 1e-6, f"resampling from p on rejection does not (TV {tv1:.4f} after the running sentence, {tv2:.4f} after held-out 800)")
key = [l.split("=", 1)[1].strip().strip('"') for l in open(os.path.join(HERE, "..", "..", ".env")) if l.startswith("ANTHROPIC_API_KEY")][0]
clean = all(key not in open(f, errors="ignore").read() for f in glob.glob(os.path.join(HERE, "**", "*"), recursive=True) if os.path.isfile(f))
check(clean, "the API key was read from .env and its string appears in no file under 04-code/ch32")
print(f"\n{sum(ok)} of {len(ok)} checkpoints pass")
assert all(ok)
