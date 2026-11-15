"""Chapter 36, 36.1: the same test with real text.  Twenty passages of 60 Tiny tokens from Through the Looking-Glass
(held out from Tiny's training), each shown twice in one window of 120; loss on the first showing against the repeat.
GPT-2 small on the same passages in its own tokenizer, for comparison.  CPU, one thread, 32-bit, seed 1865."""
from _common import *
import torch, numpy as np
torch.set_num_threads(1)
from tiny.eval import load
from _pool import corpora
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt"))
g = [int(i) for i in corpora()["glass"]]; N, T = 60, 20
rng = np.random.default_rng(SEED); starts = sorted(rng.integers(0, len(g) - N, size=T).tolist())
R = {"starts": starts}
L = np.zeros((T, 2 * N - 1))
for t, s in enumerate(starts):
    run = g[s:s + N]; ids = run + run
    with torch.no_grad(): lp = torch.log_softmax(tt.scores(ids).double(), -1)
    L[t] = -lp[torch.arange(2 * N - 1), torch.tensor(ids[1:])].numpy()
a, b = float(L[:, 1:N - 1].mean()), float(L[:, N:].mean()); R["tiny"] = [a, b]
print(f"Tiny, 20 passages of 60 tokens shown twice: first showing {a:.3f} nats per token, repeat {b:.3f}")
from transformers import AutoTokenizer, AutoModelForCausalLM
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); m = AutoModelForCausalLM.from_pretrained(CKPT("gpt2"), dtype=torch.float32).eval()
tok_t = tok(); A = []; B = []
for s in starts:
    run = tk.encode(tok_t.decode(g[s:s + N])); ids = run + run; n = len(run)
    with torch.no_grad(): lp = torch.log_softmax(m(torch.tensor([ids])).logits[0].double(), -1)
    l = -lp[torch.arange(2 * n - 1), torch.tensor(ids[1:])].numpy(); A.append(l[1:n - 1].mean()); B.append(l[n:].mean())
R["gpt2"] = [float(np.mean(A)), float(np.mean(B))]
print(f"GPT-2 small, the same passages in its tokenizer: first showing {np.mean(A):.3f}, repeat {np.mean(B):.3f}")
save_json("repeat_text", R)
