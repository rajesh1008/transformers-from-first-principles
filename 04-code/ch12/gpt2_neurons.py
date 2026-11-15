"""Chapter 12, sections 7 and 8: what GPT-2 small's feed-forward neurons read and write.
  A. Weight geometry: for each of the 3,072 neurons in each layer, the cosine between its key (column of
     c_fc) and its value (row of c_proj).  Raw tables; the normalisation gain in front is not folded in.
  B. Values read through the token table (GPT-2 ties its output layer to wte): the tokens each value
     row pushes up and pushes down, for the most selective neurons of the last two layers.
  C. Keys read through text: GPT-2 reads 1,024 tokens of the novel (opening of Chapter I, after an
     end-of-text token); the contexts on which a neuron fires hardest, and the next token there.
  D. Each layer's feed-forward block removed alone (output set to zero), and the change in loss, on the
     same text and on 1,024 tokens of Through the Looking-Glass.  Both texts are public domain and
     probably in GPT-2's training data; this measures loss, not memorisation.
GPT-2's numbers, not Tiny's.  float32.  Run from 04-code/ch12:  python3 gpt2_neurons.py  (a few minutes)"""
from _common import *
from _gpt2run import weights, norm, gelu, softmax, token_losses, decoder, H, D, L
from _bpe import encode
from tiny.corpus import load_text

W = weights(); dec = decoder()
wte = W["wte.weight"]

print("A. Key against value, cosine per neuron (3,072 per layer):")
print("   layer | median  | share > 0.5 | share < -0.5 | lowest  | highest")
geo = []
for l in range(L):
    K = W[f"h.{l}.mlp.c_fc.weight"]          # 768 x 3072: column j is neuron j's key
    V = W[f"h.{l}.mlp.c_proj.weight"]        # 3072 x 768: row j is neuron j's value
    c = (K.T * V).sum(1) / (np.linalg.norm(K, axis=0) * np.linalg.norm(V, axis=1))
    geo.append(c)
    print(f"   {l:>5} | {np.median(c):>7.3f} | {np.mean(c > 0.5):>11.3f} | {np.mean(c < -0.5):>12.3f} | {c.min():>7.3f} | {c.max():>7.3f}")

print("\nB. Values through the token table (top tokens pushed up / down), most selective neurons:")
def vocab_view(l, top=4):
    V = W[f"h.{l}.mlp.c_proj.weight"]
    kurt = np.zeros(V.shape[0]); ups = {}; downs = {}
    for s in range(0, V.shape[0], 256):
        Z = V[s:s + 256] @ wte.T                                  # 256 x 50,257
        Zc = Z - Z.mean(1, keepdims=True)
        kurt[s:s + 256] = (Zc ** 4).mean(1) / (Zc ** 2).mean(1) ** 2
        for j in range(len(Z)):
            o = np.argpartition(-Z[j], 8)[:8]; ups[s + j] = o[np.argsort(-Z[j][o])]
            o = np.argpartition(Z[j], 8)[:8]; downs[s + j] = o[np.argsort(Z[j][o])]
    return kurt, ups, downs
views = {}
for l in (10, 11):
    kurt, ups, downs = vocab_view(l)
    views[l] = (kurt, ups, downs)
    print(f"  layer {l}: median kurtosis of value-through-vocabulary scores {np.median(kurt):.1f}")
    for j in np.argsort(-kurt)[:4]:
        print(f"    neuron {l}.{j:<5} kurtosis {kurt[j]:>7.1f} | up: {[dec(t) for t in ups[j][:5]]} | down: {[dec(t) for t in downs[j][:5]]}")

alice = load_text()
glass = open(os.path.join(HERE, "..", "data", "looking_glass_gutenberg_raw.txt"), encoding="utf-8").read()
def opening(txt, marker, n=1024):
    i = txt.index(marker); return [50256] + encode(txt[i:i + 12000].replace("\r", ""))[:n - 1]
A_ids, G_ids = opening(alice, "Alice was beginning"), opening(glass, "One thing was certain")

def forward(ids, drop_ff=None, keep=()):
    """GPT-2 forward; drop_ff = layer whose feed-forward output is set to zero; keep = layers whose hidden
    activations are returned."""
    x = W["wte.weight"][ids] + W["wpe.weight"][:len(ids)]; n = len(ids); hid = {}
    mask = np.tril(np.ones((n, n), dtype=bool))
    for l in range(L):
        p = f"h.{l}."
        a = norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        q, k, v = (t.reshape(n, H, D).transpose(1, 0, 2) for t in np.split(a @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"], 3, axis=-1))
        s = np.where(mask, q @ k.transpose(0, 2, 1) / np.float32(np.sqrt(D)), np.float32(-1e9))
        x = x + (softmax(s) @ v).transpose(1, 0, 2).reshape(n, H * D) @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        m = norm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        h = gelu(m @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
        if l in keep: hid[l] = h
        if l != drop_ff:
            x = x + h @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
    return x, hid

xa, hid = forward(A_ids, keep=tuple(range(L)))
base_a = token_losses(xa, A_ids).mean()
print(f"\nC. Keys through text: Alice, Chapter I opening, {len(A_ids)} tokens; loss {base_a:.3f}")
print("   share of neuron activations above 0 and above 1, by layer:",
      [(round(float(np.mean(hid[l] > 0)), 3), round(float(np.mean(hid[l] > 1)), 4)) for l in range(L)])
def ctx(i, before=8):
    return "".join(dec(t) for t in A_ids[max(1, i - before):i + 1]).replace("\n", " ")
for l in (10, 11):
    kurt, ups, downs = views[l]
    for j in np.argsort(-kurt)[:4]:
        col = hid[l][:-1, j]                                   # positions with a next token
        top = np.argsort(-col)[:3]
        hits = sum(A_ids[i + 1] in set(ups[j][:8].tolist()) for i in np.argsort(-col)[:20])
        print(f"   neuron {l}.{j}: strongest on " + " / ".join(f"[{ctx(i)}] -> {dec(A_ids[i + 1])!r} ({col[i]:.2f})" for i in top))
        print(f"      of its 20 strongest positions, next token among the 8 it pushes up most: {hits}")

sent = [50256] + encode(" the cat sat on the mat")
xs, hs = forward(sent, keep=(5,))
print(f"\n   the running sentence {[dec(t) for t in sent[1:]]}: at ' mat', layer 5 neurons above 0: {int((hs[5][-1] > 0).sum())} of 3,072; "
      f"above 1: {int((hs[5][-1] > 1).sum())}; strongest: {[int(j) for j in np.argsort(-hs[5][-1])[:3]]}")

xg, _ = forward(G_ids); base_g = token_losses(xg, G_ids).mean()
print(f"\nD. Each layer's feed-forward block removed alone (output zero): change in loss")
print(f"   baseline: Alice {base_a:.3f} | Looking-Glass {base_g:.3f}")
for l in range(L):
    xr, _ = forward(A_ids, drop_ff=l); xq, _ = forward(G_ids, drop_ff=l)
    print(f"   layer {l:>2}: Alice {token_losses(xr, A_ids).mean() - base_a:+.3f} | Looking-Glass {token_losses(xq, G_ids).mean() - base_g:+.3f}")
