"""Chapter 29, section 29.7: precision when loading.  GPT-2 small's 32-bit tables cast to BF16 and to FP16 and run with the
book's forward pass in that format (CPU, one thread): file size, teacher-forced score differences on the running sentence,
and greedy decoding of 24 tokens from 12 held-out Carroll prompts against the 32-bit run (how many prompts diverge, and
where).  Seed 1865 for the prompt draw."""
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import load_gpt2 as LG
from _common import *
import torch
from _pool import corpora
torch.set_num_threads(1)
tk = LG.tk; W32 = dict(LG.W)
c = corpora(); t = tok()
text = t.decode([int(i) for i in c["held_b"]]); g = tk.encode(text)
starts = np.random.default_rng(SEED).choice(len(g) - 40, size=12, replace=False)
prompts = [g[s:s + 16] for s in sorted(starts)]
def run(dtype):
    LG.W = {k: v.to(dtype) for k, v in W32.items()}
    try:
        with torch.no_grad():
            S = LG.forward(tk.encode(SENTENCE)).float()
            outs = []
            for p in prompts:
                ids = list(p)
                for _ in range(24):
                    ids.append(int(LG.forward(ids)[-1].float().argmax()))
                outs.append(ids[len(p):])
        return S, outs
    finally:
        LG.W = W32
S32, G32 = run(torch.float32)
lp32 = torch.log_softmax(S32, -1)
out = {}
for name, dt in [("bf16", torch.bfloat16), ("fp16", torch.float16)]:
    S, Gs = run(dt)
    d = float((torch.log_softmax(S, -1) - lp32).abs().max())
    same_top = bool((S.argmax(-1) == S32.argmax(-1)).all())
    div = [next((i for i, (a, b) in enumerate(zip(x, y)) if a != b), None) for x, y in zip(Gs, G32)]
    n_div = sum(v is not None for v in div)
    print(f"{name}: weights {sum(v.numel() for v in W32.values()) * 2:,} bytes as stored (32-bit {sum(v.numel() for v in W32.values()) * 4:,}); "
          f"running sentence max |log-prob diff| {d:.3e}, same top token everywhere {same_top}; greedy 24 tokens from 12 prompts: "
          f"{n_div} of 12 diverge from 32-bit, first differing positions {sorted(v for v in div if v is not None)}", flush=True)
    out[name] = dict(max_logp=d, same_top=same_top, diverged=n_div, first=[v for v in div])
m = []
for p, gg in zip(prompts, G32):
    ids = list(p)
    for k in range(24):
        with torch.no_grad():
            s = torch.sort(LG.forward(ids)[-1], descending=True).values
        m.append(float(s[0] - s[1])); ids.append(gg[k])
print(f"32-bit greedy steps: top-two score margin median {np.median(m):.3f}, share of steps with margin below 0.1: {np.mean(np.array(m) < 0.1):.3f}")
out["margins"] = dict(median=float(np.median(m)), below01=float(np.mean(np.array(m) < 0.1)))
save_json("precision", out)
