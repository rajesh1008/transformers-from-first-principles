"""Chapter 25, section 25.10: reads outputs/judge.json and outputs/bench_right.npz (no API call) and prints the hosted
model's results beside Tiny's and the counting baselines' on the same 100 Looking-Glass cloze items, and the pattern of
the pairwise judge's answers.  Writes outputs/judge_summary.txt."""
import _common
import os, json, collections
import numpy as np
from _common import HERE
d = json.load(open(os.path.join(HERE, "outputs", "judge.json")))
R = np.load(os.path.join(HERE, "outputs", "bench_right.npz"))
P1, P2 = d["P1"], d["P2"]
out = []
out.append("same 100 Looking-Glass cloze items (sum scoring for Tiny and the counts):")
for k in ("625_steps", "1250_steps", "2500_steps", "unigram", "bigram"):
    out.append(f"  {k:12s} {R[k][:100].mean():.2f}")
for o in ("shuffled", "reversed"):
    ps = [p for p in P1 if p["order"] == o]
    out.append(f"  hosted, {o:8s} {np.mean([p['answer'] == p['right'] for p in ps]):.2f}; replies naming the source instead of a letter {sum(p['answer'] == '?' for p in ps)}")
wrong_both = sum(P1[2 * i]["answer"] != P1[2 * i]["right"] and P1[2 * i + 1]["answer"] != P1[2 * i + 1]["right"] for i in range(100))
out.append(f"  hosted, wrong in both orders: {wrong_both} items; right letters in the prompts {dict(sorted(collections.Counter(p['right'] for p in P1).items()))}")
pairs = collections.Counter((P2[2 * i]["answer"], P2[2 * i + 1]["answer"]) for i in range(60))
out.append("pairwise judge, answers (true sentence first, true sentence second) over 60 passages: " +
           ", ".join(f"{a}/{b}: {n}" for (a, b), n in sorted(pairs.items())))
txt = "\n".join(out)
print(txt)
open(os.path.join(HERE, "outputs", "judge_summary.txt"), "w").write(txt + "\n")
