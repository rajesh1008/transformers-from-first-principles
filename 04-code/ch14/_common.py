"""Shared set-up for Chapter 14's scripts (run from 04-code/ch14).  NumPy only.
Reference machine: the book's build machine (CPU, 8 cores), NumPy 2.5.4; timings carry their date."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "ch04"))   # _gpt2, the GPT-2 weight reader from Chapter 4
sys.path.insert(0, os.path.join(HERE, "..", "ch09"))   # _gpt2run (GPT-2 small in NumPy) and _bpe (its tokenizer)
import numpy as np
np.set_printoptions(suppress=True)
SEED = 1865
SENTENCE = "the cat sat on the mat"
TINY_BPE = os.path.join(HERE, "..", "tiny", "tiny_bpe_2048.txt")

def r3(v, k=3):
    return [round(float(x), k) for x in np.ravel(v)]

def length(v):
    return float(np.linalg.norm(v))

def cos(a, b):
    return float(np.dot(np.ravel(a), np.ravel(b)) / (np.linalg.norm(a) * np.linalg.norm(b)))

def recorded(name, measure):
    """Timings are measured once on the build machine and recorded; --measure times yours."""
    path = os.path.join(HERE, "outputs", name + ".json")
    if "--measure" in sys.argv or not os.path.exists(path):
        result = measure()
        result["measured"] = __import__("datetime").date.today().isoformat()
        json.dump(result, open(path, "w"), indent=1)
    return json.load(open(path))

def best_of(f, k=5):
    import time
    ts = []
    for _ in range(k):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts) * 1000

def tok():
    from tiny.tokenizer import load_bpe
    return load_bpe(TINY_BPE)

def sentence_ids(t=None):
    t = t or tok()
    return t.encode(SENTENCE)

def novel_ids(n=None, t=None):
    from tiny.corpus import load_text
    t = t or tok()
    ids = t.encode(load_text())
    return ids if n is None else ids[:n]

def piece(t, i):
    """A token as text; a lone byte that is not valid text on its own is shown as <0xNN>."""
    i = int(i)
    b = t.vocab.get(i) if hasattr(t, "vocab") else None
    if isinstance(b, bytes):
        try:
            txt = b.decode("utf-8")
            if txt.isprintable() or txt.strip() == "" and txt != "":
                return txt
            return "".join(f"<0x{c:02X}>" for c in b)
        except UnicodeDecodeError:
            return "".join(f"<0x{c:02X}>" for c in b)
    return t.decode([i])
