"""Chapter 29, section 29.7 follow-up: why FP16 did worse than BF16.  Measures the largest entry of GPT-2's residual stream on
the running sentence and the largest squared deviation inside a layer norm (FP16's largest finite number is 65,504), then
reruns the FP16 and BF16 checks with layer norm computed in 32-bit (as libraries do)."""
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import load_gpt2 as LG
from _common import *
import torch
torch.set_num_threads(1)
tk = LG.tk; W32 = dict(LG.W); ids = tk.encode(SENTENCE)
mx = {"stream": 0.0, "sq": 0.0}
orig_ln = LG.ln
def ln_probe(x, p, eps=LG.eps):
    m = x.float().mean(-1, keepdim=True)
    mx["stream"] = max(mx["stream"], float(x.float().abs().max())); mx["sq"] = max(mx["sq"], float(((x.float() - m) ** 2).max()))
    return orig_ln(x, p, eps)
LG.ln = ln_probe
with torch.no_grad():
    R = torch.log_softmax(LG.forward(ids), -1)
LG.ln = orig_ln
print(f"32-bit: largest |entry| entering a layer norm {mx['stream']:.1f}; largest squared deviation {mx['sq']:.1f} (FP16 max 65,504)")
def ln32(x, p, eps=LG.eps):
    return orig_ln(x.float(), p, eps).to(x.dtype)
for name, dt in [("bf16", torch.bfloat16), ("fp16", torch.float16)]:
    LG.W = {k: (v.float() if ("ln" in k) else v.to(dt)) for k, v in W32.items()}
    for label, fn in [("layer norm in low precision", orig_ln), ("layer norm in 32-bit", ln32)]:
        LG.ln = fn
        if fn is orig_ln:
            LG.W = {k: v.to(dt) for k, v in W32.items()}
        else:
            LG.W = {k: (v.float() if (".ln" in "." + k or k.startswith("ln_")) else v.to(dt)) for k, v in W32.items()}
        with torch.no_grad():
            S = LG.forward(ids).float()
        d = float((torch.log_softmax(S, -1) - R).abs().max())
        print(f"  {name}, {label:<28}: max |log-prob diff| {d:.3e}; finite {bool(torch.isfinite(S).all())}; same top token everywhere {bool((S.argmax(-1) == R.argmax(-1)).all())}")
    LG.ln = orig_ln; LG.W = W32
