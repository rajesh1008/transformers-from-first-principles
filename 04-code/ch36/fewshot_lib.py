"""Shared prompt builders for fewshot.py and fewshot_top.py (Chapter 36)."""

from _common import *
import torch, numpy as np
torch.set_num_threads(1); torch.manual_seed(SEED)
from transformers import AutoTokenizer, AutoModelForCausalLM
PAIRS = [("hot","cold"),("big","small"),("up","down"),("fast","slow"),("happy","sad"),("light","dark"),("rich","poor"),
         ("hard","soft"),("long","short"),("high","low"),("wet","dry"),("early","late"),("full","empty"),("strong","weak"),
         ("loud","quiet"),("thick","thin"),("good","bad"),("true","false"),("first","last"),("black","white"),("day","night"),
         ("old","young"),("open","closed"),("left","right")]
FORMATS = {"arrow": ("{} ->", " {}\n"), "io": ("Input: {}\nOutput:", " {}\n\n"), "space": ("{}", " {}\n")}
MIXED = [("{} ->", " {}\n"), ("Input: {}\nOutput:", " {}\n\n"), ("{} =", " {}\n"), ("Q: {} A:", " {}\n"), ("{}:", " {}\n"),
         ("{} |", " {}\n"), ("{} =>", " {}\n"), ("word {} opposite", " {}\n")]
def build(qi, k, rng, fmt="arrow", labels="right"):
    others = [i for i in range(len(PAIRS)) if i != qi]; demo = rng.choice(others, size=k, replace=False).tolist() if k else []
    outs = [PAIRS[i][1] for i in demo]
    if labels == "shuffled" and k > 1:
        perm = rng.permutation(k)
        while k > 1 and any(perm[j] == j for j in range(k)): perm = rng.permutation(k)
        outs = [outs[j] for j in perm]
    parts = []; segs = []  # (text, kind)
    for j, i in enumerate(demo):
        q, a = (MIXED[j % len(MIXED)] if fmt == "mixed" else FORMATS[fmt])
        segs += [(q.format(PAIRS[i][0]).replace(PAIRS[i][0], "\x00"), "sep"), (PAIRS[i][0], "demo_in"), (a.format(outs[j]), "demo_out")]
    q, _ = MIXED[k % len(MIXED)] if fmt == "mixed" else FORMATS[fmt]
    segs += [(q.format(PAIRS[qi][0]), "query")]
    return segs
def encode(tk, segs, bos):
    """Tokenize segment by segment so every token keeps a label (query segment: the word and its separator)."""
    text = ""; spans = []
    for s, kind in segs:
        if kind == "sep":
            pre, post = s.split("\x00"); spans.append((len(text), len(text) + len(pre), "sep")); text += pre; continue
        if kind == "demo_in":
            spans.append((len(text), len(text) + len(s), "demo_in")); text += s
            continue
        if kind == "demo_out":
            # the separator part that followed the input word was in the "sep" template after the placeholder
            pass
        spans.append((len(text), len(text) + len(s), kind)); text += s
    return text, spans
def prompt_text(segs):
    out = ""
    for s, kind in segs:
        if kind == "sep": continue
        out += s
    return out
def run_model(name):
    tk = AutoTokenizer.from_pretrained(CKPT(name))
    m = AutoModelForCausalLM.from_pretrained(CKPT(name), dtype=torch.float32, attn_implementation="eager").eval()
    def score(text, ans, attn=False):
        ids = tk.encode(text); a = tk.encode(" " + ans)[0]
        with torch.no_grad(): o = m(torch.tensor([ids]), output_attentions=attn)
        lp = torch.log_softmax(o.logits[0, -1].double(), -1)
        return int(lp.argmax() == a), float(lp[a]), ids, (o.attentions if attn else None)
    R = {}
    def sweep(tag, k, fmt="arrow", labels="right"):
        rng = np.random.default_rng(SEED); acc = []; lps = []
        for qi in range(len(PAIRS)):
            segs = build(qi, k, rng, fmt, labels); text = full_text(segs, fmt, k)
            c, l, _, _ = score(text, PAIRS[qi][1]); acc.append(c); lps.append(l)
        R[tag] = dict(acc=float(np.mean(acc)), lp=float(np.mean(lps)))
        print(f"{name:8s} {tag:22s} accuracy {np.mean(acc):.3f}  mean log p(right first token) {np.mean(lps):7.3f}", flush=True)
    for k in (0, 1, 2, 4, 8): sweep(f"arrow k={k}", k)
    for fmt in ("io", "space", "mixed"): sweep(f"{fmt} k=4", 4, fmt)
    sweep("arrow k=4 shuffled", 4, "arrow", "shuffled"); sweep("arrow k=8 shuffled", 8, "arrow", "shuffled")
    # where the last row looks, k = 4, arrow format: share of attention by kind of token, mean over heads, layers and queries
    rng = np.random.default_rng(SEED); kinds = ["first", "demo_in", "arrow", "demo_out", "newline", "query_word", "query_arrow"]
    tot = None; per_head = None
    for qi in range(len(PAIRS)):
        segs = build(qi, 4, rng); text = full_text(segs, "arrow", 4)
        _, _, ids, A = score(text, PAIRS[qi][1], attn=True)
        lab = label_tokens(tk, ids)
        A = torch.stack([a[0, :, -1, :] for a in A])  # L x H x n
        sh = torch.stack([A[:, :, [i for i, x in enumerate(lab) if x == kd]].sum(-1) if kd in lab else torch.zeros(A.shape[:2]) for kd in kinds])  # K x L x H
        tot = sh if tot is None else tot + sh
    tot = tot / len(PAIRS)
    print(f"{name}: k=4 arrow prompts, share of the last row's attention by token kind (mean over all heads):", flush=True)
    print("   " + "  ".join(f"{kd} {float(tot[i].mean()):.3f}" for i, kd in enumerate(kinds)), flush=True)
    L, H = tot.shape[1:]; flat = tot[kinds.index("demo_out")].flatten(); top = flat.argsort(descending=True)[:6]
    print("   heads putting most attention on the examples' answers (layer.head: share on answers, on query word):", flush=True)
    print("   " + "   ".join(f"{int(t)//H}.{int(t)%H}: {float(flat[t]):.3f}, {float(tot[kinds.index('query_word')].flatten()[t]):.3f}" for t in top), flush=True)
    R["attention_by_kind"] = {kd: float(tot[i].mean()) for i, kd in enumerate(kinds)}
    R["top_answer_heads"] = [(f"{int(t)//H}.{int(t)%H}", float(flat[t])) for t in top]
    # by layer: share on answers
    bylayer = tot[kinds.index("demo_out")].mean(-1)
    print("   share on the examples' answers by layer: " + " ".join(f"{float(x):.2f}" for x in bylayer), flush=True)
    R["answers_by_layer"] = [float(x) for x in bylayer]
    return R
def full_text(segs, fmt, k):
    out = ""; j = 0; i = 0
    while i < len(segs):
        s, kind = segs[i]
        if kind == "sep":
            q, a = (MIXED[j % len(MIXED)] if fmt == "mixed" else FORMATS[fmt])
            word = segs[i + 1][0]; out += q.format(word) + segs[i + 2][0]; i += 3; j += 1; continue
        out += s; i += 1
    return out
def label_tokens(tk, ids):
    """Label each token of an arrow-format prompt: first, demo_in, arrow, demo_out, newline, query_word, query_arrow."""
    toks = [tk.decode([t]) for t in ids]; lab = []; state = "in"; nl = sum(1 for t in toks if "\n" in t)
    seen_nl = 0
    for i, t in enumerate(toks):
        if i == 0 and "->" not in t: pass
        if "\n" in t: lab.append("newline"); seen_nl += 1; state = "in"; continue
        last = seen_nl == nl
        if "->" in t or t.strip() in ("-", ">"): lab.append("query_arrow" if last else "arrow"); state = "out"; continue
        if last: lab.append("query_word")
        else: lab.append("demo_out" if state == "out" else "demo_in")
    lab[0] = "first"
    return lab
