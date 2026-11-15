"""Chapter 32, 32.7a: verifying a tree of guesses in one pass.  Prompt lookup with up to B distinct continuations (the
most recent earlier matches of the final two tokens), each k = 4 long, scored in one target pass with a tree mask
(each guess sees the cache, the last confirmed token and the guesses before it on its own branch) and its own position.
Keep the branch with the longest agreeing run.  B = 1 is the chain of 32.4.  Greedy, 64 new tokens after the four
prompts of 32.4; checked against plain greedy.  CPU, one thread, 32-bit; timings best of 3, 11 Nov 2026."""
from _common import *
import torch, time, math
torch.set_num_threads(1)
from tiny.load import GPT2
from tiny.cache import new_cache, greedy
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2")); g = held_text_ids(tk)
THE = tk.encode(" the")[0]

def tree_pass(m, cache, toks, pos, parent):
    """One pass over new nodes toks at positions pos; parent[i] = index of node i's parent among new nodes (-1 = none).
    Node i sees the whole cache and its ancestors.  Appends the nodes' keys and values to cache; returns scores."""
    W, H, d = m.W, m.H, m.d; T = len(toks); hd = d // H
    anc = torch.zeros(T, T, dtype=torch.bool)
    for i in range(T):
        j = i
        while j >= 0: anc[i, j] = True; j = parent[j]
    x = W["wte.weight"][torch.as_tensor(toks)] + W["wpe.weight"][torch.as_tensor(pos)]
    for l in range(m.L):
        p = f"h.{l}."
        q, k, v = (m._ln(x, p + "ln_1") @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]).split(d, -1)
        q, k, v = (z.view(T, H, hd).transpose(0, 1) for z in (q, k, v))
        P = 0 if cache[l] is None else cache[l][0].shape[1]
        if cache[l] is not None: k = torch.cat([cache[l][0], k], 1); v = torch.cat([cache[l][1], v], 1)
        cache[l] = (k, v)
        mask = torch.cat([torch.ones(T, P, dtype=torch.bool), anc], 1)
        s = (q @ k.transpose(-1, -2) / math.sqrt(hd)).masked_fill(~mask, float("-inf"))
        o = (torch.softmax(s, -1) @ v).transpose(0, 1).reshape(T, d)
        x = x + o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        f = m._gelu(m._ln(x, p + "ln_2") @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
        x = x + f @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
    return m._ln(x, "ln_f") @ W["wte.weight"].T

def candidates(ids, k, B):
    out = []
    for i in range(len(ids) - 3, -1, -1):
        if ids[i] == ids[-2] and ids[i + 1] == ids[-1]:
            c = ids[i + 2:i + 2 + k]
            if c:
                c = c + [c[-1]] * (k - len(c))
                if c not in out: out.append(c)
            if len(out) == B: break
    return out or [[THE] * k]

def spec_tree(prompt, n, k, B):
    ids = list(prompt); cache = new_cache(gm); st = dict(passes=0, rows=0, accepted=0, branches=0)
    with torch.no_grad():
        s = gm.forward(ids, cache, 0)[-1]; st["passes"] += 1; ids.append(int(s.argmax()))
        while len(ids) - len(prompt) < n:
            br = candidates(ids, k, B); st["branches"] += len(br)
            n0 = len(ids) - 1                                 # cache holds positions 0 .. n0 - 1
            toks = [ids[-1]]; pos = [n0]; parent = [-1]; where = []
            for b in br:
                idx = []
                for j, t in enumerate(b):
                    parent.append(0 if j == 0 else len(toks) - 1); toks.append(t); pos.append(n0 + 1 + j); idx.append(len(toks) - 1)
                where.append(idx)
            S = tree_pass(gm, cache, toks, pos, parent); st["passes"] += 1; st["rows"] += len(toks)
            top = S.argmax(-1).tolist(); best = (-1, None)
            for b, idx in zip(br, where):
                mm, prev = 0, 0
                while mm < k and b[mm] == top[prev]: prev = idx[mm]; mm += 1
                if mm > best[0]: best = (mm, b, idx, prev)
            mm, b, idx, prev = best
            keep = [n0 + 0] + [n0 + i for i in idx[:mm]]       # root and accepted nodes, as cache positions
            sel = torch.tensor(list(range(n0)) + keep)
            for l, c in enumerate(cache): cache[l] = (c[0][:, sel], c[1][:, sel])
            ids += b[:mm] + [top[prev]]; st["accepted"] += mm
    return ids[:len(prompt) + n], st

def best(f, r=3):
    ts = []
    for _ in range(r):
        t0 = time.perf_counter(); o = f(); ts.append(time.perf_counter() - t0)
    return min(ts), o
print("Prompt lookup as a chain (B 1) and as a tree of up to B branches, k = 4, greedy, 64 new tokens per prompt")
prompts = {"separator (token 0)": g[0:64], "held-out 800": g[800:864], "held-out 1600": g[1600:1664], "held-out 2400": g[2400:2464]}
res = {}
for name, p in prompts.items():
    tpl, (ref, _) = best(lambda: greedy(gm, p, 64, True))
    line = f"  {name:20s}: plain {tpl:5.2f} s"
    for B in (1, 3):
        t, (ids, st) = best(lambda: spec_tree(p, 64, 4, B))
        res[f"{name} B{B}"] = dict(same=ids == ref, tpp=64 / (st["passes"] - 1), rows=st["rows"] / (st["passes"] - 1),
                                   br=st["branches"] / (st["passes"] - 1), time=t, speedup=tpl / t)
        line += (f" | B {B}: same {ids == ref}, {64 / (st['passes'] - 1):4.2f} tokens/pass, {st['rows'] / (st['passes'] - 1):4.1f} rows/pass "
                 f"({st['branches'] / (st['passes'] - 1):3.1f} branches), {t:5.2f} s ({tpl / t:4.2f} x)")
    print(line, flush=True)
save_json("spec_tree", res)
