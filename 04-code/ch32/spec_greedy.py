"""Chapter 32, 32.1 to 32.4 (partial): greedy speculative decoding on GPT-2 small with three same-tokenizer drafts.
Part A: what verifying k tokens in one pass costs against one token (cached, context 256).
Part B: three drafts, k = 4, 64 new tokens after four held-out prompts: GPT-2's own first L layers + final norm + tied
output (an untrained early exit), a prompt-lookup draft (copy what followed the last matching bigram in the context), and
the broken draft that always guesses ' the'.  Output must equal plain greedy decoding; report acceptance and time.
CPU, one thread, 32-bit, greedy; timings best of 3, 11 Nov 2026."""
from _common import *
import torch, time, math
torch.set_num_threads(1)
from tiny.load import GPT2
from tiny.cache import new_cache, greedy
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2")); g = held_text_ids(tk)
THE = tk.encode(" the")[0]

def fwd_layers(m, ids, cache, start, L):
    """GPT2.forward truncated after L layers, then ln_f and the tied output (an early exit)."""
    W, H, d = m.W, m.H, m.d; T = len(ids); hd = d // H
    x = W["wte.weight"][torch.as_tensor(ids)] + W["wpe.weight"][start:start + T]
    for l in range(L):
        p = f"h.{l}."
        q, k, v = (m._ln(x, p + "ln_1") @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]).split(d, -1)
        q, k, v = (z.view(T, H, hd).transpose(0, 1) for z in (q, k, v))
        if cache[l] is not None:
            k = torch.cat([cache[l][0], k], 1); v = torch.cat([cache[l][1], v], 1)
        cache[l] = (k, v); S = k.shape[1]
        mask = torch.ones(T, S, dtype=torch.bool).tril(S - T)
        s = (q @ k.transpose(-1, -2) / math.sqrt(hd)).masked_fill(~mask, float("-inf"))
        o = (torch.softmax(s, -1) @ v).transpose(0, 1).reshape(T, d)
        x = x + o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        f = m._gelu(m._ln(x, p + "ln_2") @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
        x = x + f @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
    return m._ln(x, "ln_f") @ W["wte.weight"].T

def trim(cache, n):
    for l, c in enumerate(cache):
        if c is not None: cache[l] = (c[0][:, :n], c[1][:, :n])

class EarlyExit:
    def __init__(self, L): self.L = L; self.name = f"first {L} layers"
    def start(self, ids):
        self.cache = [None] * self.L
        with torch.no_grad(): fwd_layers(gm, ids[:-1], self.cache, 0, self.L)
        self.n = len(ids) - 1
    def propose(self, ids, k):
        trim(self.cache, self.n); out = []; cur = ids[self.n:]
        with torch.no_grad():
            for _ in range(k):
                s = fwd_layers(gm, cur, self.cache, self.n, self.L)[-1]; self.n += len(cur)
                t = int(s.argmax()); out.append(t); cur = [t]
        self.n -= 0
        return out
    def accept(self, n_ok): self.n = n_ok        # positions whose keys are valid = tokens confirmed
class Lookup:
    name = "prompt lookup"
    def start(self, ids): pass
    def propose(self, ids, k):
        for i in range(len(ids) - 3, -1, -1):
            if ids[i] == ids[-2] and ids[i + 1] == ids[-1]:
                c = ids[i + 2:i + 2 + k]
                return c + [c[-1]] * (k - len(c)) if c else [THE] * k
        return [THE] * k
    def accept(self, n): pass
class The:
    name = "always ' the'"
    def start(self, ids): pass
    def propose(self, ids, k): return [THE] * k
    def accept(self, n): pass

def speculative(prompt, n, draft, k=4):
    ids = list(prompt); cache = new_cache(gm); passes = 0; acc = 0; prop = 0
    with torch.no_grad():
        s = gm.forward(ids, cache, 0)[-1]; passes += 1
        ids.append(int(s.argmax()))                      # the target's own first token
        draft.start(ids)
        while len(ids) - len(prompt) < n:
            guess = draft.propose(ids, k); prop += k
            S = gm.forward([ids[-1]] + guess, cache, len(ids) - 1); passes += 1
            tgt = S.argmax(-1).tolist()                  # target's choice after each prefix
            m = 0
            while m < k and guess[m] == tgt[m]: m += 1
            acc += m
            ids += guess[:m] + [tgt[m]]                  # accepted guesses plus one token from the target
            trim(cache, len(ids) - 1)
            draft.accept(len(ids) - 1)
    return ids[:len(prompt) + n], dict(passes=passes, accepted=acc, proposed=prop)

def best(f, r=3):
    ts = []
    for _ in range(r):
        t0 = time.perf_counter(); out = f(); ts.append(time.perf_counter() - t0)
    return min(ts), out

print("Part A: one cached pass over k new tokens at context 256 (time relative to k = 1)")
base = new_cache(gm)
with torch.no_grad(): gm.forward(g[:256], base, 0)
resA = {}
for k in (1, 2, 3, 5, 9, 17):
    def one():
        c = [(a.clone(), b.clone()) for a, b in base]
        with torch.no_grad(): gm.forward(g[256:256 + k], c, 256)
    t, _ = best(one, 5); resA[k] = t
    print(f"  k {k:2d}: {t * 1000:6.1f} ms ({t / resA[1]:4.2f} x one token)", flush=True)

print("\nPart B: 64 new tokens after four held-out prompts of 64 tokens; k = 4 guesses per target pass")
prompts = [g[i:i + 64] for i in (0, 800, 1600, 2400)]
out = {"A": resA, "B": {}}
plain = []
tp = 0.0
for p in prompts:
    t, (ref, _) = best(lambda: greedy(gm, p, 64, True)); tp += t; plain.append(ref)
print(f"  plain cached greedy: {tp:6.2f} s for 4 x 64 tokens ({256 / tp:5.1f} tokens/s); 64 target passes per prompt")
for draft in (EarlyExit(2), EarlyExit(6), Lookup(), The()):
    tt = 0.0; same = True; P = A = Q = 0
    for p, ref in zip(prompts, plain):
        t, (ids, st) = best(lambda: speculative(p, 64, draft, 4)); tt += t
        same &= ids == ref; P += st["passes"]; A += st["accepted"]; Q += st["proposed"]
    print(f"  {draft.name:16s}: same as greedy {same}; accepted {A}/{Q} guesses ({A / Q:.3f}); target passes {P} for 256 tokens "
          f"({256 / P:4.2f} tokens per pass); {tt:6.2f} s ({tp / tt:4.2f} x plain)", flush=True)
    out["B"][draft.name] = dict(same=same, acc=A / Q, passes=P, time=tt, speedup=tp / tt)
out["plain_time"] = tp
print(f"\n  first prompt, plain greedy continues: {tk.decode(plain[0][64:96])!r}")
save_json("spec_greedy", out)
