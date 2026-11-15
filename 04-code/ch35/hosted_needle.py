"""Chapter 35, 35.5: the pass-key test on a hosted model, at lengths the book's models cannot reach.  Model: Claude Haiku 5.5
(model id claude-haiku-5-5) through Anthropic's Messages API, default sampling, extended thinking off, at most 20 output tokens.
The haystack is Through the Looking-Glass as plain text (repeated from the start if a length needs more); the key sentence
" The pass key is NNNNN." is inserted after the first full stop at or after the chosen depth; the question asks for the five
digits only.  Lengths in characters 8,000 / 64,000 / 256,000 / 512,000; depths 0, 0.25, 0.5, 0.75, 1.0; two keys per cell (seed
1865); plus two rival conditions at 256,000 characters.  Input token counts are the API's own (usage.input_tokens).
The key is read from ../../.env into this process only; it is never printed or written.  Writes outputs/hosted_needle.json."""
from _common import *
import re, json, time, random, urllib.request, concurrent.futures as cf
MODEL = "claude-haiku-5-5"
def _key():
    for l in open(os.path.join(HERE, "..", "..", ".env")):
        if l.startswith("ANTHROPIC_API_KEY="): return l.split("=", 1)[1].strip().strip('"')
KEY = _key()
def ask(prompt):
    body = json.dumps(dict(model=MODEL, max_tokens=20, thinking={"type": "disabled"},
                           messages=[dict(role="user", content=prompt)])).encode()
    err = ""
    for attempt in range(5):
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST",
                                         headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=180))
            return "".join(b.get("text", "") for b in r["content"]).strip(), r.get("usage", {}).get("input_tokens")
        except Exception as e:
            err = type(e).__name__ + (" " + str(e.code) if hasattr(e, "code") else ""); time.sleep(3 + 5 * attempt)
    return "ERROR " + err, None
G = glass_text()
def hay(n):
    s = G
    while len(s) < n: s += "\n\n" + G
    return s[:n]
def insert(text, frac, sentence):
    at = int(frac * len(text))
    if at >= len(text): return text + sentence
    j = text.find(".", at)
    j = len(text) if j < 0 else j + 1
    return text[:j] + sentence + text[j:]
rng = random.Random(SEED); keys = [str(rng.randint(10000, 99999)) for _ in range(2)]; rivals = [str(rng.randint(10000, 99999)) for _ in range(2)]
Q = "\n\nWhat is the pass key? Reply with the five digits only."
jobs = []
for n in (8000, 64000, 256000, 512000):
    for d in (0.0, 0.25, 0.5, 0.75, 1.0):
        for k in keys: jobs.append((n, f"depth {d}", k, None, insert(hay(n), d, f" The pass key is {k}.") + Q))
for k, r in zip(keys, rivals):
    h = hay(256000)
    jobs.append((256000, "rival at 0.25, key at 0.75", k, r, insert(insert(h, 0.75, f" The pass key is {k}."), 0.25, f" The pass key is {r}.") + Q))
    jobs.append((256000, "key at 0.25, rival at 0.75", k, r, insert(insert(h, 0.75, f" The pass key is {r}."), 0.25, f" The pass key is {k}.") + Q))
with cf.ThreadPoolExecutor(4) as ex: answers = list(ex.map(lambda j: ask(j[4]), jobs))
R = {"model": MODEL, "keys_count": len(keys), "rows": []}
cells = {}
for (n, cond, k, r, _), (a, ntok) in zip(jobs, answers):
    got = re.findall(r"\d{5}", a); g = got[0] if got else a[:20]
    R["rows"].append(dict(chars=n, cond=cond, correct=g == k, rival=(r is not None and g == r), reply=g, input_tokens=ntok))
    c = cells.setdefault((n, cond), [0, 0, 0, []]); c[0] += g == k; c[1] += 1; c[2] += (r is not None and g == r); c[3].append(ntok)
for (n, cond), (ok, tot, rv, toks) in cells.items():
    print(f"{n:7,} characters (~{max(t or 0 for t in toks):,} input tokens)  {cond:28s} correct {ok} of {tot}" + (f"; rival's key {rv} of {tot}" if "rival" in cond else ""), flush=True)
save_json("hosted_needle", R)
