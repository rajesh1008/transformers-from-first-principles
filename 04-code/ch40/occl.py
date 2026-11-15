"""Chapter 40, 40.6 and 40.10: replace one part of the teacher-forced prompt by the token ' and' and count how many of the 30 taught
verbs are still the model's first choice at the site (Chapter 38's fine-tuned Tiny).  Parts: the question's animal name; the ' The' that opens
the answer; the answer's animal name before its last token (only animals whose name is two or more tokens); "Tiny:" ; the whole question."""
from lib import *
import numpy as np
torch.set_num_threads(1)
exec(open("facts_common.py").read())
AND = T.encode(" and")[0]
def still(a, idx):
    e = P[a]; ids = list(e["ids"][:e["site"] + 1])
    for i in idx: ids[i] = AND
    with torch.no_grad(): return int(forward(ft, ids)["logits"][0, -1].argmax()) == e["verb"]
parts = {
 "the question's animal name": lambda e: e["span"],
 "' The' opening the answer": lambda e: [e["site"] - len(T.encode(" " + e["a"]))],
 "the answer's animal name, tokens before the last": lambda e: list(range(e["site"] - len(T.encode(" " + e["a"])) + 1, e["site"])),
 "'Tiny:' (T, iny, :)": lambda e: [e["q"], e["q"] + 1, e["q"] + 2],
 "the whole question (What did the ... do?)": lambda e: list(range(4, e["q"] - 1)),
}
R = {}
for k, f in parts.items():
    use = [a for a in names if f(P[a])]
    r = sum(still(a, f(P[a])) for a in use); R[k] = [r, len(use)]
    print(f"  replace {k:50s}: verb still first choice for {r} of {len(use)}")
save_json("occl", R)
