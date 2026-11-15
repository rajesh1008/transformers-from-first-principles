"""Chapter 26, section 26.8: summary of outputs/repeat_hosted.json (no API call).  Prints the temperature-0 refusal and, for
each prompt, the number of distinct replies in 20 and the commonest replies (cut at 50 characters)."""
import _common
import os, json, re
from collections import Counter
from _common import HERE
d = json.load(open(os.path.join(HERE, "outputs", "repeat_hosted.json")))
m = re.search(r'"message":"(.*?)"', d["temperature0"])
print("temperature 0:", "refused, " + (m.group(1) if m else d["temperature0"][:80]) if d["temperature0"].startswith("ERROR") else "accepted")
for k, v in d["replies"].items():
    c = Counter(v)
    top = "; ".join(f"{n} x {t[:50] + ('...' if len(t) > 50 else '')!r}" for t, n in c.most_common(3))
    print(f"{k}: {len(c)} distinct replies in {len(v)}; commonest {top}")
first = Counter(r.strip()[:1] for r in d["replies"]["P2"])
print("P2 first characters:", dict(first), "| replies naming the book:", sum("Looking-Glass" in r for r in d["replies"]["P2"]))
