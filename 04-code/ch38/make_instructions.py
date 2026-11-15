"""Chapter 38, 38.2: a tiny instruction set for Tiny, built by script from a small invented world and from Tiny's own
Carroll pool (no model call, no outside data).  Three kinds of pair:
  fact     "What did the X do?" -> "The X <verb> on the <thing>."  30 invented facts (the cat's is the running sentence);
           10 animals appear once in training, 10 four times, 10 twelve times (the cat is in the last group).
  repeat   "Repeat: <5 to 8 words of the pool>" -> the same words.
  finish   "Finish the line: <6 words of the pool>" -> the next 6 words.
Training set 500 pairs (170 fact, 165 repeat, 165 finish), shuffled.  Test set: every fact question once (30), the same
facts asked another way, "Tell me what the X did." (30), 40 unseen repeats and 40 unseen finishes.  Seed 1865.
Writes outputs/instructions.json."""
from _common import *
import numpy as np, re
from _pool import corpora
rng = np.random.default_rng(SEED)
ANIMALS = "cat dog mouse owl frog pig duck crab snail lion goat hen fox bee crow eel toad mole lamb hare dove wren seal bear deer wolf swan bat ant cow".split()
VERBS = "sat slept stood sang danced waited hid fell".split()
THINGS = "mat hat rug box log step wall bench chair stone".split()
facts = {}
for a in ANIMALS:
    facts[a] = ("sat", "mat") if a == "cat" else (VERBS[rng.integers(len(VERBS))], THINGS[rng.integers(len(THINGS))])
order = [a for a in ANIMALS if a != "cat"]; rng.shuffle(order)
reps = {a: 1 for a in order[:10]}; reps.update({a: 4 for a in order[10:20]}); reps.update({a: 12 for a in order[20:29]}); reps["cat"] = 12
ans = lambda a: f"The {a} {facts[a][0]} on the {facts[a][1]}."
text = tok().decode([int(i) for i in corpora()["pool"]])
words = [w for w in re.findall(r"[A-Za-z][A-Za-z']*[,.;]?", text)]
def spans(n, k):
    out = set()
    while len(out) < k:
        s = int(rng.integers(0, len(words) - n)); out.add(" ".join(words[s:s + n]))
    return list(out)
rep = spans(7, 205); fin = spans(12, 205)
train = [dict(kind="fact", animal=a, prompt=f"What did the {a} do?", response=ans(a)) for a in ANIMALS for _ in range(reps[a])]
train += [dict(kind="repeat", prompt=f"Repeat: {p}", response=p) for p in rep[:165]]
train += [dict(kind="finish", prompt=f"Finish the line: {' '.join(p.split()[:6])}", response=" ".join(p.split()[6:])) for p in fin[:165]]
order = rng.permutation(len(train)); train = [train[i] for i in order]
test = [dict(kind="fact", animal=a, reps=reps[a], prompt=f"What did the {a} do?", response=ans(a)) for a in ANIMALS]
test += [dict(kind="fact-reworded", animal=a, reps=reps[a], prompt=f"Tell me what the {a} did.", response=ans(a)) for a in ANIMALS]
test += [dict(kind="repeat", prompt=f"Repeat: {p}", response=p) for p in rep[165:]]
test += [dict(kind="finish", prompt=f"Finish the line: {' '.join(p.split()[:6])}", response=" ".join(p.split()[6:])) for p in fin[165:]]
save_json("instructions", dict(train=train, test=test, facts={a: list(v) for a, v in facts.items()}, reps=reps))
print(f"train {len(train)} pairs; test {len(test)}; example pairs:")
for e in train[:6]: print(f"  {e['kind']:7s} {e['prompt']!r} -> {e['response']!r}")
print("facts seen 12 times:", sorted(a for a in ANIMALS if reps[a] == 12))
