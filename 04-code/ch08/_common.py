import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "ch04"))   # _gpt2, the small GPT-2 reader from Chapter 4
import numpy as np
np.set_printoptions(suppress=True)
WORDS = ["the", "cat", "sat", "on", "the", "mat"]
X = np.array([[1, 0], [0, 1], [1, 1], [1, -1], [1, 0], [-1, 1]], dtype=float)   # Chapter 4's toy rows
W_q = np.array([[1, 0], [0, 2]], dtype=float)      # keeps the first number, doubles the second
W_k = np.array([[0, 1], [1, 0]], dtype=float)      # swaps the two numbers
W_v = np.array([[1, 0], [1, 1]], dtype=float)      # new first = first + second; new second = second
SENTENCE = "the cat sat on the mat"
TINY_BPE = os.path.join(HERE, "..", "tiny", "tiny_bpe_2048.txt")

def table(M, rows=WORDS, cols=WORDS, fmt="{:>7.3f}", width=7):
    head = "        " + "".join(f"{c:>{width}}" for c in cols)
    lines = [head]
    for r, row in zip(rows, M):
        lines.append(f"{r:>6}  " + "".join(fmt.format(v) if np.isfinite(v) else f"{'-inf':>{width}}" for v in row))
    return "\n".join(lines)

import json
def recorded(name, measure):
    """Timings are measured once on the build machine and recorded; --measure times yours."""
    path = os.path.join(HERE, "outputs", name + ".json")
    if "--measure" in sys.argv or not os.path.exists(path):
        result = measure()
        result["measured"] = __import__("datetime").date.today().isoformat()
        json.dump(result, open(path, "w"), indent=1)
    return json.load(open(path))
