"""Shared set-up for Chapter 46 scripts (copied from ../ch37; run from 04-code/ch46).  CPU, one thread, 32-bit, seed 1865.
Weights are read from ../ch29/ckpt/ (fetched in Chapter 29; not in the code zip)."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "ch29"))
SEED = 1865
SENTENCE = "the cat sat on the mat"
CKPT = lambda n: os.path.join(HERE, "..", "ch29", "ckpt", n)
def save_json(name, obj):
    json.dump(obj, open(os.path.join(HERE, "outputs", name + ".json"), "w"), indent=1)
def load_json(name):
    return json.load(open(os.path.join(HERE, "outputs", name + ".json")))
TINY_BPE = os.path.join(HERE, "..", "tiny", "tiny_bpe_2048.txt")
def held_text_ids(tk):
    """GPT-2 ids of Chapter 22's held-out Carroll text (as Chapter 30 used)."""
    from _pool import corpora
    from tiny.tokenizer import load_bpe
    tt = load_bpe(TINY_BPE); c = corpora()
    return tk.encode(tt.decode([int(i) for i in c["held_b"]]))
def tok():
    from tiny.tokenizer import load_bpe
    return load_bpe(TINY_BPE)
def glass_text():
    """Through the Looking-Glass as plain text (held out from Tiny's training pool; GPT-2 and SmolLM2 have probably read it)."""
    from _pool import corpora
    return tok().decode([int(i) for i in corpora()["glass"]])
