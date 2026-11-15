"""Chapter 48, 48.10: how often a lower-case word beginning 'mat' in the training books is the word 'mat' itself.  -> outputs/mat48.txt"""
import os, re
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
t = open(os.path.join(HERE, "corpus", "train.txt"), encoding="utf-8").read()
c = Counter(re.findall(r"(?<![A-Za-z])mat([A-Za-z]*)", t)); n = sum(c.values())
out = f"words beginning 'mat' (lower case): {n:,}; the word 'mat' itself: {c['']:,}; commonest: " + ", ".join(f"mat{k} {v:,}" for k, v in c.most_common(6) if k)
print(out); open(os.path.join(HERE, "outputs", "mat48.txt"), "w").write(out + "\n")
