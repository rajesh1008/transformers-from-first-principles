"""Chapter 38, session 30 (the tryit): the same instruction set with reworded questions in training.  Reads
outputs/instructions.json and adds "Tell me what the X did." for 15 of the 30 animals (as many times as the fact itself
is taught), drawn at random with seed 1865.  Test set: the original 140 prompts, with the reworded fact questions split
into the 15 whose wording was trained (fact-reworded-seen) and the 15 whose wording was not (fact-reworded-unseen), plus a
third wording never trained for any animal, "What was the X doing?" (fact-new-wording).  Writes outputs/instructions_reword.json."""
from _common import *
import numpy as np
D = load_json("instructions"); rng = np.random.default_rng(SEED)
animals = list(D["facts"]); seen = set(rng.choice(animals, size=15, replace=False).tolist())
ans = lambda a: f"The {a} {D['facts'][a][0]} on the {D['facts'][a][1]}."
extra = [dict(kind="fact-reworded", animal=a, prompt=f"Tell me what the {a} did.", response=ans(a)) for a in animals if a in seen for _ in range(D["reps"][a])]
train = D["train"] + extra; order = rng.permutation(len(train)); train = [train[i] for i in order]
test = []
for e in D["test"]:
    e = dict(e)
    if e["kind"] == "fact-reworded": e["kind"] = "fact-reworded-seen" if e["animal"] in seen else "fact-reworded-unseen"
    test.append(e)
test += [dict(kind="fact-new-wording", animal=a, reps=D["reps"][a], prompt=f"What was the {a} doing?", response=ans(a)) for a in animals]
save_json("instructions_reword", dict(train=train, test=test, facts=D["facts"], reps=D["reps"], seen=sorted(seen)))
print(f"train {len(train)} pairs ({len(extra)} reworded added); test {len(test)}; reworded wording trained for: {sorted(seen)}")
print("cat in the trained-wording group:", "cat" in seen)
