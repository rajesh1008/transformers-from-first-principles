"""Chapter 48, 48.10: how many of the words the model writes are words of the training books, by checkpoint.
For each saved checkpoint of the main run: 20 samples of 120 tokens at temperature 0.8 (seeds 1 to 20) from the running sentence,
with the cached loop of eval48.py; every maximal run of letters (apostrophes inside allowed) after the prompt is a word, and a word
counts as known if it occurs, ignoring case, in corpus/train.txt.  -> outputs/words48.json"""
import os, sys, re, json, glob
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from tiny.capstone.model import Capstone
from tokenizers import Tokenizer
torch.set_num_threads(1)
C = os.path.join(HERE, "corpus"); tk = Tokenizer.from_file(os.path.join(C, "tok8192.json"))
W = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")
known = {w.lower() for w in W.findall(open(os.path.join(C, "train.txt"), encoding="utf-8").read())}
PROMPT = "the cat sat on the mat"; pid = tk.encode(PROMPT).ids; R = {}
def sample(m, seed, n=120):
    g = torch.Generator().manual_seed(seed); ids = list(pid)
    with torch.no_grad():
        for _ in range(n):
            lg = m(torch.tensor([ids[-256:]]))[0, -1]
            ids.append(int(torch.multinomial(torch.softmax(lg / 0.8, -1), 1, generator=g)))
    return tk.decode(ids)[len(PROMPT):]
for f in sorted(glob.glob(os.path.join(HERE, "ckpt", "main_step*.pt")), key=lambda f: int(f.split("step")[-1][:-3])):
    ck = torch.load(f, map_location="cpu"); m = Capstone(**ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
    words = [w for s in range(1, 21) for w in W.findall(sample(m, s))]
    k = sum(w.lower() in known for w in words); R[ck["step"]] = dict(words=len(words), known=k, share=k / len(words))
    print(f"step {ck['step']}: {len(words)} words in 20 samples, {k} in the training books ({k / len(words):.3f})", flush=True)
json.dump(dict(known_vocabulary=len(known), steps=R), open(os.path.join(HERE, "outputs", "words48.json"), "w"), indent=1)
