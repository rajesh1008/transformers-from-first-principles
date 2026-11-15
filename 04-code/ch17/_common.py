"""Shared set-up for Chapter 17's scripts (run from 04-code/ch17).  NumPy, and PyTorch (CPU) for autograd.
Reference machine: the book's build machine (CPU, 8 cores), NumPy 2.5.4, PyTorch 2.15.0+cpu; timings carry
their date and are one thread unless stated.  Seed 1865 everywhere a random number is drawn."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")          # one thread, set before NumPy or torch load
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "ch15"))   # hand.py: the hand Tiny of Chapter 15
sys.path.insert(0, os.path.join(HERE, "..", "ch14"))
import numpy as np
np.set_printoptions(suppress=True)
SEED = 1865
SENTENCE = "the cat sat on the mat"

def r3(v, k=3):
    return [round(float(x), k) for x in np.ravel(v)]

def save_json(name, obj):
    json.dump(obj, open(os.path.join(HERE, "outputs", name + ".json"), "w"), indent=1)

def load_json(name):
    return json.load(open(os.path.join(HERE, "outputs", name + ".json")))

TINY_BPE = os.path.join(HERE, "..", "tiny", "tiny_bpe_2048.txt")

def tok():
    from tiny.tokenizer import load_bpe
    return load_bpe(TINY_BPE)

def sentence_ids(t=None):
    return (t or tok()).encode(SENTENCE)

def novel_ids(t=None):
    from tiny.corpus import load_text
    return (t or tok()).encode(load_text())

def heldout_ids(t=None):
    from tiny.corpus import load_heldout
    return (t or tok()).encode(load_heldout())

def best_of(f, k=5):
    import time
    ts = []
    for _ in range(k):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts) * 1000
