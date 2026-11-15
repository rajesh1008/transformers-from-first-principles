"""Chapter 29, section 29.5: GPT-2 on the held-out text in short windows, 110 GPT-2 tokens (about the characters of Tiny's
129-token windows at 0.313 against 0.366 tokens per character), so that both models see about the same context."""
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import load_gpt2 as LG
from _common import *
import torch, math
from _pool import corpora
torch.set_num_threads(1)
tk = LG.tk; c = corpora(); t = tok()
g = tk.encode(t.decode([int(i) for i in c["held_b"]]))
for ctx in (110, 1024):
    tot, chars, n = 0.0, 0, 0
    for a in range(0, len(g) - 1, ctx):
        w = g[a:a + ctx]
        if len(w) < 2: break
        with torch.no_grad():
            lp = torch.log_softmax(LG.forward(w).double(), -1)
        s = -lp[torch.arange(len(w) - 1), torch.tensor(w[1:])]
        tot += float(s.sum()); n += len(w) - 1; chars += len(tk.decode(w[1:]))
    print(f"GPT-2, windows of {ctx:4d}: mean surprise {tot / n:.4f} nats per token over {n:,} tokens; {tot / math.log(2) / chars:.4f} bits per character")
