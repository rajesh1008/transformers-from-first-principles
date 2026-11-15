"""Chapter 8 cameo: real GPT-2 small heads on the running sentence.  GPT-2's weights are read with
Chapter 4's NumPy reader; the parts of each block that later chapters build (normalisation, Chapter 13;
the feed-forward block, Chapter 12) are included here only so that layers after the first get the
right input.  Comparison only: these are not Tiny's numbers.
Run from 04-code/ch08:  python3 gpt2_head.py"""
from _common import *
from _gpt2 import tensor
from tiny.attention import softmax, causal_mask

ids = [1169, 3797, 3332, 319, 262, 2603]          # GPT-2's ids for the running sentence (Chapter 2)
pieces = ["the", "␣cat", "␣sat", "␣on", "␣the", "␣mat"]
H, D = 12, 64

def norm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

x = tensor("wte.weight")[ids] + tensor("wpe.weight")[:6]
n = len(ids)
all_shares = []
for layer in range(12):
    p = f"h.{layer}."
    a = norm(x, tensor(p + "ln_1.weight"), tensor(p + "ln_1.bias"))
    qkv = a @ tensor(p + "attn.c_attn.weight") + tensor(p + "attn.c_attn.bias")   # (6, 2304)
    q, k, v = np.split(qkv, 3, axis=-1)
    heads_out = []
    layer_shares = []
    for h in range(H):
        qh, kh, vh = (t[:, h * D:(h + 1) * D] for t in (q, k, v))
        s = qh @ kh.T / np.sqrt(D)
        A = softmax(np.where(causal_mask(n), s, -np.inf))
        layer_shares.append(A)
        heads_out.append(A @ vh)
    all_shares.append(layer_shares)
    o = np.concatenate(heads_out, -1) @ tensor(p + "attn.c_proj.weight") + tensor(p + "attn.c_proj.bias")
    x = x + o
    m = norm(x, tensor(p + "ln_2.weight"), tensor(p + "ln_2.bias"))
    x = x + gelu(m @ tensor(p + "mlp.c_fc.weight") + tensor(p + "mlp.c_fc.bias")) @ tensor(p + "mlp.c_proj.weight") + tensor(p + "mlp.c_proj.bias")
S = np.array(all_shares)                           # (12 layers, 12 heads, 6, 6)
np.save(os.path.join(HERE, "outputs", "gpt2_shares.npy"), S)   # for solutions.py, exercise 13

print("GPT-2 small, running sentence ids", ids)
print("query and key width per head:", D, "-> scores divided by sqrt(64) = 8")
print("\nlayer 0, head 0:")
print(table(S[0, 0], rows=pieces, cols=pieces))
l, h = np.unravel_index(np.argmax(S[:, :, 2, 1]), S.shape[:2])
print(f"\nthe head where ␣sat gives the most to ␣cat: layer {l}, head {h}, share {S[l, h, 2, 1]:.3f}")
print(table(S[l, h], rows=pieces, cols=pieces))
print("\nshare ␣sat gives to ␣cat, averaged over the 12 heads of each layer:")
print("  " + "  ".join(f"L{i}:{S[i, :, 2, 1].mean():.2f}" for i in range(12)))
first = S[:, :, 1:, 0].mean(axis=-1)               # rows 1-5, share on the first token
print("\nshare given to the first token 'the' (rows 1 to 5), averaged per layer:")
print("  " + "  ".join(f"L{i}:{first[i].mean():.2f}" for i in range(12)))
print(f"heads (of 144) whose rows 1-5 put more than half their share on the first token: {(first > 0.5).sum()}")
print(f"for comparison, an even split over the seats each row can see gives the first token {np.mean([1 / (r + 1) for r in range(1, 6)]):.2f} on average")

# a check that the forward pass is right: GPT-2's own next-token guesses after each prefix
xf = norm(x, tensor("ln_f.weight"), tensor("ln_f.bias"))
logits = xf @ tensor("wte.weight").T
import json, urllib.request
VOCAB = os.path.join(os.path.expanduser("~/.cache/tfp/gpt2"), "vocab.json")
if not os.path.exists(VOCAB):
    urllib.request.urlretrieve("https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json", VOCAB)
inv = {i: t for t, i in json.load(open(VOCAB)).items()}       # GPT-2 writes a leading space as 'Ġ'
print("\ncheck: GPT-2's top three next tokens after 'the cat sat on the':",
      [inv[int(i)].replace("\u0120", "\u2423") for i in np.argsort(logits[4])[::-1][:3]])
print("check: the vocabulary gives the running sentence back:", [inv[i].replace("\u0120", "\u2423") for i in ids])
