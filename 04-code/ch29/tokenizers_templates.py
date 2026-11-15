"""Chapter 29, section 29.9: tokenizers and chat templates.  GPT-2 small and SmolLM2-135M (base and Instruct) through
tiny/load.py.  Part A: special tokens in the three files.  Part B: scoring with and without a start-of-text token.
Part C: the literal string '<|endoftext|>' in user text.  Part D: a trailing space.  Part E: SmolLM2-135M-Instruct
answering five questions under four prompt formats (greedy, 40 tokens, our own forward pass).  Part F: round trips.
32-bit, one thread; no draws (greedy only)."""
from _common import *
import torch, math, json
torch.set_num_threads(1)
from tiny.load import GPT2, Llama, compare, render_chatml
from transformers import AutoTokenizer
from _pool import corpora
out = {}
D = lambda n: os.path.join(HERE, "ckpt", n)
tg, ts, ti = AutoTokenizer.from_pretrained(D("gpt2")), AutoTokenizer.from_pretrained(D("smollm2")), AutoTokenizer.from_pretrained(D("smollm2i"))
gm = GPT2.from_dir(D("gpt2"))

print("Part A: special tokens")
for name, tk in [("GPT-2", tg), ("SmolLM2-135M", ts), ("SmolLM2-135M-Instruct", ti)]:
    sp = {t: tk.convert_tokens_to_ids(t) for t in tk.all_special_tokens}
    gc = json.load(open(os.path.join(D({"GPT-2": "gpt2", "SmolLM2-135M": "smollm2", "SmolLM2-135M-Instruct": "smollm2i"}[name]), "generation_config.json")))
    print(f"  {name:<22} vocabulary {len(tk):,}; special tokens {len(sp)}: {dict(list(sp.items())[:4])}{' ...' if len(sp) > 4 else ''}; "
          f"generation_config bos {gc.get('bos_token_id')} eos {gc.get('eos_token_id')}; encode('the cat') adds a start token: "
          f"{tk.encode('the cat')[0] in sp.values()}")

print("\nPart B: GPT-2 scored with and without <|endoftext|> (50256) in front")
ids = tg.encode(SENTENCE)
def surprise(seq, first):                   # mean surprise of tokens seq[first:] (each predicted from what precedes it)
    lp = torch.log_softmax(gm.scores(seq).double(), -1)
    return float(-lp[torch.arange(first - 1, len(seq) - 1), torch.tensor(seq[first:])].mean())
a, b = surprise(ids, 1), surprise([50256] + ids, 2)
print(f"  running sentence, tokens 2 to 6: no start token {a:.3f} nats; with start token {b:.3f}")
c = corpora(); t = tok()
g = tg.encode(t.decode([int(i) for i in c["held_b"]]))
st = np.random.default_rng(SEED).choice(len(g) - 40, size=50, replace=False)
A, B, A1, B1 = [], [], [], []
for s in sorted(st):
    w = g[s:s + 32]
    lp0 = torch.log_softmax(gm.scores(w).double(), -1); lp1 = torch.log_softmax(gm.scores([50256] + w).double(), -1)
    s0 = -lp0[torch.arange(0, 31), torch.tensor(w[1:])]; s1 = -lp1[torch.arange(1, 32), torch.tensor(w[1:])]
    A.append(float(s0.mean())); B.append(float(s1.mean())); A1.append(float(s0[:3].mean())); B1.append(float(s1[:3].mean()))
A, B, A1, B1 = map(np.array, (A, B, A1, B1))
print(f"  50 held-out Carroll windows of 32 tokens (seed {SEED}), tokens 2 to 32: no start {A.mean():.3f}, with start {B.mean():.3f}; "
      f"paired difference {np.mean(A - B):+.3f} (se {np.std(A - B, ddof=1) / math.sqrt(50):.3f}); better with start in {(B < A).sum()} of 50")
print(f"  tokens 2 to 4 only: no start {A1.mean():.3f}, with start {B1.mean():.3f}")
out["B"] = dict(sent=[a, b], no=float(A.mean()), yes=float(B.mean()), first3=[float(A1.mean()), float(B1.mean())], wins=int((B < A).sum()))

print("\nPart C: the literal text '<|endoftext|>' inside a prompt")
txt = "The cat sat on the mat.<|endoftext|>The"
e1 = tg.encode(txt); e2 = tg(txt, split_special_tokens=True)["input_ids"]
print(f"  default encode: {len(e1)} ids, contains 50256: {50256 in e1}; split_special_tokens=True: {len(e2)} ids {e2[6:]}, contains 50256: {50256 in e2}")
cat = tg.encode(" cat")[0]
for lab, e in [("as one control id", e1), ("as ordinary text", e2), ("no marker ('The cat sat on the mat. The')", tg.encode("The cat sat on the mat. The"))]:
    p = torch.softmax(gm.scores(e)[-1].double(), -1)
    top = torch.argsort(-p)[:5]
    print(f"  {lab:<44} next ' cat' share {float(p[cat]):.4f} (rank {int((p > p[cat]).sum()) + 1}); top 5 " + ", ".join(f"{tg.decode([int(j)])!r} {float(p[j]):.3f}" for j in top))
out["C"] = dict(n1=len(e1), n2=len(e2))

print("\nPart D: a trailing space")
for pr in ["the cat sat on the", "the cat sat on the "]:
    e = tg.encode(pr); p = torch.softmax(gm.scores(e)[-1].double(), -1); top = torch.argsort(-p)[:6]
    H = float(-(p * torch.log(p.clamp_min(1e-300))).sum())
    print(f"  {pr!r:<24} ids end {e[-2:]}; entropy {H:.3f}; top 6 " + ", ".join(f"{tg.decode([int(j)])!r} {float(p[j]):.3f}" for j in top))
pm = torch.softmax(gm.scores(tg.encode("the cat sat on the "))[-1].double(), -1)
words = [i for i in range(50257) if tg.decode([i]).startswith(" ") and tg.decode([i])[1:2].isalpha()]
print(f"  after the trailing space, total share of tokens that begin with a space and a letter (' floor' etc.): {float(pm[words].sum()):.4f}")
print(f"  'mat' without a space is {tg.encode('mat')}, with one {tg.encode(' mat')}")

print("\nPart E: SmolLM2-135M-Instruct, five questions, four prompt formats, greedy 40 tokens with our own forward pass")
im = Llama.from_dir(D("smollm2i"))
msgs = [[{"role": "user", "content": q}] for q in ["What is the capital of France?", "Name three colours.",
        "What is 2 + 3?", "Write one sentence about a cat.", "Who wrote Alice's Adventures in Wonderland?"]]
from transformers import LlamaForCausalLM
ref = LlamaForCausalLM.from_pretrained(D("smollm2i"), dtype=torch.float32).eval()
r0 = ti.apply_chat_template(msgs[0], tokenize=False, add_generation_prompt=True)
print(f"  library template == hand-written render_chatml: {r0 == render_chatml(msgs[0])}")
print(f"  rendered: {r0!r}")
e0 = ti.encode(r0)
with torch.no_grad():
    cmpd = compare(im.scores(e0), ref(torch.tensor([e0])).logits[0])
print(f"  {len(e0)} tokens ({len(ti.encode(msgs[0][0]['content']))} of them the question); own forward vs reference on it: {cmpd}")
fmts = {"ChatML template (correct)": lambda m: render_chatml(m),
        "template, no generation prompt": lambda m: render_chatml(m, add_generation_prompt=False),
        "plain question, no template": lambda m: m[0]["content"],
        "another family's template ([INST])": lambda m: f"<s>[INST] {m[0]['content']} [/INST]"}
res = {}
for lab, f in fmts.items():
    stops, lens, firsts = 0, [], []
    for m in msgs:
        e = ti.encode(f(m)); n0 = len(e); stop = False
        for _ in range(40):
            nx = int(im.scores(e)[-1].argmax()); e.append(nx)
            if nx == 2:
                stop = True; break
        stops += stop; lens.append(len(e) - n0); firsts.append(ti.decode(e[n0:n0 + 14]))
    res[lab] = dict(stops=stops, lens=lens)
    print(f"  {lab:<36} ended with <|im_end|> within 40 tokens: {stops} of 5; tokens written {lens}")
    for q, fr in zip(msgs[:2], firsts[:2]):
        print(f"      {q[0]['content'][:28]!r:<32} -> {fr!r}")
out["E"] = res

print("\nPart F: round trips, decode(encode(s)) == s")
ss = ["naïve café", "two  spaces", "tab\there", "日本語", "🐈 on a mat", "the cat sat on the mat"]
for name, tk in [("GPT-2", tg), ("SmolLM2", ts)]:
    print(f"  {name:<8} " + "; ".join(f"{s!r} {len(tk.encode(s))} ids {'ok' if tk.decode(tk.encode(s)) == s else 'CHANGED'}" for s in ss))
tt_ = tok(); row = []
for s in ss:
    try:
        e = tt_.encode(s); row.append(f"{s!r} {len(e)} ids {'ok' if tt_.decode(e) == s else 'CHANGED'}")
    except Exception as ex:
        row.append(f"{s!r} error {type(ex).__name__}")
print("  Tiny     " + "; ".join(row))
save_json("tokenizers_templates", out)
