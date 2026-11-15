"""Shared set-up for Chapter 30 scripts (run from 04-code/ch30).  CPU, one thread, 32-bit, seed 1865.
Weights are read from ../ch29/ckpt/ (fetched in Chapter 29; not in the code zip)."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "ch29"))
import numpy as np
SEED = 1865
SENTENCE = "the cat sat on the mat"
CKPT = lambda n: os.path.join(HERE, "..", "ch29", "ckpt", n)
def save_json(name, obj):
    json.dump(obj, open(os.path.join(HERE, "outputs", name + ".json"), "w"), indent=1)
TINY_BPE = os.path.join(HERE, "..", "tiny", "tiny_bpe_2048.txt")
def tok():
    from tiny.tokenizer import load_bpe
    return load_bpe(TINY_BPE)
