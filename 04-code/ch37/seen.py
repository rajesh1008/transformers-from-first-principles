"""Chapter 37, 37.3: what the model has and has not seen.
A  Tiny on text it trained on (20 windows of its Carroll pool) against text it never saw (20 windows of Through the
   Looking-Glass): loss, top-1 accuracy and mean top-1 confidence.
B  GPT-2 small and SmolLM2-135M on "The capital of X is" for 15 well-known and 15 less-known countries (the author's
   split, made before the run), and for 6 invented countries: a greedy continuation of 10 tokens, whether it contains
   the capital, and the mean share of the generated tokens (the model's confidence in what it wrote).
   (A first version scored only the greedy first token; both models put it on ' the' for 49 of 60 queries, so the
   question was rerun with continuations; see RUN.txt.)
CPU, one thread, 32-bit, seed 1865."""
from _common import *
import torch, numpy as np
torch.set_num_threads(1); torch.manual_seed(SEED)
from transformers import AutoTokenizer, AutoModelForCausalLM
from tiny.eval import load
from _pool import corpora
R = {}
tt = load(os.path.join(HERE, "..", "ch24", "outputs", "ckpt", "wsd_1e-3_2000_b2000.pt")); c = corpora()
for name, ids in (("pool (trained on)", [int(i) for i in c["pool"]]), ("Looking-Glass (never seen)", [int(i) for i in c["glass"]])):
    st = np.random.default_rng(SEED).integers(0, len(ids) - 129, size=20); L = []; A = []; C = []
    for s in st:
        ch = ids[s:s + 129]
        with torch.no_grad(): lp = torch.log_softmax(tt.scores(ch)[:-1].double(), -1)
        tg = torch.tensor(ch[1:]); L.append(float(-lp[torch.arange(128), tg].mean()))
        A.append(float((lp.argmax(-1) == tg).double().mean())); C.append(float(lp.max(-1).values.exp().mean()))
    print(f"Tiny on {name}: loss {np.mean(L):.4f} nats, top-1 accuracy {np.mean(A):.4f}, mean top-1 confidence {np.mean(C):.4f}", flush=True)
    R[f"tiny {name}"] = dict(loss=float(np.mean(L)), acc=float(np.mean(A)), conf=float(np.mean(C)))
WELL = [("France","Paris"),("Japan","Tokyo"),("Italy","Rome"),("Egypt","Cairo"),("Russia","Moscow"),("Spain","Madrid"),("Germany","Berlin"),
        ("Greece","Athens"),("England","London"),("China","Beijing"),("Ireland","Dublin"),("Austria","Vienna"),("Portugal","Lisbon"),("Poland","Warsaw"),("Thailand","Bangkok")]
LESS = [("Mongolia","Ulaanbaatar"),("Bhutan","Thimphu"),("Laos","Vientiane"),("Malawi","Lilongwe"),("Suriname","Paramaribo"),("Eritrea","Asmara"),
        ("Tajikistan","Dushanbe"),("Burkina Faso","Ouagadougou"),("Moldova","Chisinau"),("Kyrgyzstan","Bishkek"),("Gabon","Libreville"),("Belize","Belmopan"),
        ("Lesotho","Maseru"),("Tuvalu","Funafuti"),("Benin","Porto-Novo")]
FAKE = ["Zubrovia", "Veltrania", "Oskaria", "Marundi Kel", "Talvessa", "Quorvath"]
for name, lab in (("gpt2", "GPT-2 small"), ("smollm2", "SmolLM2-135M")):
    tk = AutoTokenizer.from_pretrained(CKPT(name)); m = AutoModelForCausalLM.from_pretrained(CKPT(name), dtype=torch.float32).eval()
    def gen(c):
        ids = torch.tensor([tk.encode(f"The capital of {c} is")])
        with torch.no_grad(): o = m.generate(ids, max_new_tokens=10, do_sample=False, output_scores=True, return_dict_in_generate=True, pad_token_id=tk.eos_token_id)
        new = o.sequences[0, ids.shape[1]:]; sh = [float(torch.softmax(sc[0].double(), -1)[t]) for sc, t in zip(o.scores, new)]
        return tk.decode(new).replace("\n", " ").strip(), float(np.mean(sh))
    res = {}
    for grp, items in (("well-known", WELL), ("less-known", LESS)):
        right = 0; cr = []; cw = []; rows = []
        for cn, city in items:
            t, v = gen(cn); ok = city.lower()[:5] in t.lower(); right += ok
            (cr if ok else cw).append(v); rows.append((cn, t, round(v, 3), bool(ok)))
        res[grp] = dict(right=right, conf_right=float(np.mean(cr)) if cr else None, conf_wrong=float(np.mean(cw)) if cw else None, rows=rows)
        print(f"{lab} {grp}: capital named in {right} of 15; mean share of generated tokens when right {np.mean(cr) if cr else float('nan'):.3f}, when not {np.mean(cw) if cw else float('nan'):.3f}", flush=True)
        for r in rows: print(f"   {r[0]}: {r[1]!r} ({r[2]:.3f}{', right' if r[3] else ''})", flush=True)
    rows = []
    for cn in FAKE:
        t, v = gen(cn); rows.append((cn, t, round(v, 3)))
    res["invented"] = rows
    print(f"{lab} invented countries:", flush=True)
    for r in rows: print(f"   {r[0]}: {r[1]!r} ({r[2]:.3f})", flush=True)
    R[lab] = res
save_json("seen", R)
