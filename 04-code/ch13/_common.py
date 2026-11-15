"""Shared set-up for Chapter 13's scripts (run from 04-code/ch13).  NumPy only.
Reference machine: the book's build machine, CPU only; timings carry their date."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "ch04"))   # _gpt2, the GPT-2 weight reader from Chapter 4
sys.path.insert(0, os.path.join(HERE, "..", "ch09"))   # _gpt2run (GPT-2 small in NumPy) and _bpe (its tokenizer)
import numpy as np
np.set_printoptions(suppress=True)
from tiny.attention import attention
from tiny.ffn import relu
SEED = 1865
WORDS = ["the", "cat", "sat", "on", "the", "mat"]
X = np.array([[1, 0], [0, 1], [1, 1], [1, -1], [1, 0], [-1, 1]], dtype=float)   # Chapter 4's toy rows
W_q = np.array([[1, 0], [0, 2]], dtype=float)      # Chapter 8's hand tables
W_k = np.array([[0, 1], [1, 0]], dtype=float)
W_v = np.array([[1, 0], [1, 1]], dtype=float)
W_1 = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)                # Chapter 12's hand block
b_1 = np.array([0, 0, -2], dtype=float)
W_2 = np.array([[0, 0.5], [0.5, 0], [-0.5, 0.5]], dtype=float)
SENTENCE = "the cat sat on the mat"
TINY_BPE = os.path.join(HERE, "..", "tiny", "tiny_bpe_2048.txt")

def hand_head(R):
    out, _ = attention(R, W_q, W_k, W_v)
    return out

def hand_block(R):
    return relu(R @ W_1 + b_1) @ W_2

def hand_layer(R0):
    """Chapter 12's layer on the toy rows: x <- x + head(x), then x <- x + block(x).  No norm yet."""
    A = R0 + hand_head(R0)
    return A, A + hand_block(A)

def r3(v, k=3):
    return [round(float(x), k) for x in np.ravel(v)]

def recorded(name, measure):
    """Timings are measured once on the build machine and recorded; --measure times yours."""
    path = os.path.join(HERE, "outputs", name + ".json")
    if "--measure" in sys.argv or not os.path.exists(path):
        result = measure()
        result["measured"] = __import__("datetime").date.today().isoformat()
        json.dump(result, open(path, "w"), indent=1)
    return json.load(open(path))

def tiny_rows(n=None):
    """Tiny's token + position rows: the running sentence, or the first n tokens of the novel."""
    from tiny.embed import Embedding, PositionTable
    from tiny.tokenizer import load_bpe
    tok = load_bpe(TINY_BPE)
    E, P = Embedding(len(tok)), PositionTable()
    if n is None:
        ids = tok.encode(SENTENCE)
    else:
        from tiny.corpus import load_text
        ids = tok.encode(load_text())[:n]
    return tok, ids, E(ids) + P(range(len(ids)))
