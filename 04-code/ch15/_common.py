"""Shared set-up for Chapter 15's scripts (run from 04-code/ch15).  NumPy only; the pencil trace in
pencil.py uses plain Python floats and the math module, nothing else.
Reference machine: the book's build machine (CPU, 8 cores), NumPy 2.5.4; no timings in this chapter."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "ch14"))
import numpy as np
np.set_printoptions(suppress=True)
SEED = 1865
SENTENCE = "the cat sat on the mat"
TINY_BPE = os.path.join(HERE, "..", "tiny", "tiny_bpe_2048.txt")

def r3(v, k=3):
    return [round(float(x), k) for x in np.ravel(v)]

def save_json(name, obj):
    json.dump(obj, open(os.path.join(HERE, "outputs", name + ".json"), "w"), indent=1)
