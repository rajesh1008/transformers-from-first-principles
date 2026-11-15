"""Chapter 25, section 25.10: a hosted model as reader and as judge (one run, 11 November 2026).
Model: Claude Haiku 5.5 (model id claude-haiku-5-5) through Anthropic's Messages API, default sampling (the API rejected a temperature
setting for this model on 11 Nov 2026, "temperature is deprecated"), extended thinking off, a one-line system prompt
asking for a single character, at most 40 output tokens.  The key is read from ../../.env into this process only; it is never printed or written.
  Part 1  the first 100 Looking-Glass items of the constructed cloze set (outputs/cloze_items.json, from bench.py), the
          passage's last 400 characters and four lettered choices, asked twice: choices in a shuffled order (seed 1865)
          and in the reverse of that order.  Accuracy in each order; how often the answer changes with the order.
  Part 2  a pairwise judge: 60 passages of Looking-Glass (400 characters ending at a full stop, seed 1865) and two
          candidate next sentences, the true one and a sentence of similar length from elsewhere in the book; asked
          which follows, once with the true sentence first and once second.
Writes outputs/judge.json (prompts' item numbers, the answers; no key)."""
import _common
import os, re, json, time, urllib.request, concurrent.futures as cf
import numpy as np
from _pool import corpora
from _common import HERE, SEED, tok
MODEL = "claude-haiku-5-5"
def _key():
    for l in open(os.path.join(HERE, "..", "..", ".env")):
        if l.startswith("ANTHROPIC_API_KEY="):
            return l.split("=", 1)[1].strip().strip('"')
KEY = _key()
SYSTEM = "You answer multiple-choice questions. Reply with the single character of your choice and nothing else."

def parse(a, allowed):
    """The chosen character: the reply itself if it is one allowed character, else the first allowed character standing
    alone (as in 'B.' or '**2**'); '?' if none.  Raw replies are kept in judge.json."""
    if a.startswith("ERROR"):
        return a
    t = a.strip().strip("*").strip()
    if t[:1] in allowed and (len(t) == 1 or not t[1].isalnum()):
        return t[:1]
    m = re.search(r"(?<![A-Za-z0-9])([" + allowed + r"])(?![A-Za-z0-9])", a)
    return m.group(1) if m else "?"

def ask(prompt):
    body = json.dumps(dict(model=MODEL, max_tokens=40, thinking={"type": "disabled"}, system=SYSTEM,
                           messages=[dict(role="user", content=prompt)])).encode()
    for attempt in range(5):
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST",
                                         headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=60))
            return "".join(b.get("text", "") for b in r["content"]).strip()
        except Exception as e:
            err = type(e).__name__ + (" " + str(e.code) if hasattr(e, "code") else "")
            time.sleep(2 + 3 * attempt)
    return "ERROR " + err

items = [it for it in json.load(open(os.path.join(HERE, "outputs", "cloze_items.json"))) if it["src"] == "glass"][:100]
rng = np.random.default_rng(SEED)
L = "ABCD"
P1 = []
for i, it in enumerate(items):
    words = [it["word"]] + it["distractors"]
    perm = list(rng.permutation(4))
    for order_name, order in (("shuffled", perm), ("reversed", perm[::-1])):
        opts = [words[j] for j in order]
        ans = L[opts.index(it["word"])]
        prompt = ("The passage below stops just before a word. Which of the four words comes next in the original text? "
                  "Answer with the letter only.\n\nPassage: ..." + it["context_tail"] + " ____\n\n" +
                  "\n".join(f"{L[k]}. {w}" for k, w in enumerate(opts)))
        P1.append(dict(item=i, order=order_name, right=ans, prompt=prompt))
T = tok(); c = corpora()
text = T.decode(list(map(int, c["glass"])))
sents = [m for m in re.finditer(r"[A-Z][^.?!]{30,160}[.?!]", text)]
P2 = []
pick = rng.choice(np.arange(5, len(sents) - 1), size=60, replace=False)
for i, j in enumerate(sorted(pick)):
    passage = text[max(0, sents[j].start() - 400):sents[j].start()].strip()
    true = sents[j].group(0)
    ln = len(true)
    pool = [k for k in range(len(sents)) if abs(k - j) > 50 and abs(len(sents[k].group(0)) - ln) < 0.2 * ln]
    other = sents[int(rng.choice(pool))].group(0)
    for first in ("true", "other"):
        a, b = (true, other) if first == "true" else (other, true)
        prompt = ("Which of the two candidate sentences comes next after the passage in the original text? Answer 1 or 2 only.\n\n"
                  f"Passage: ...{passage}\n\nCandidate 1: {a}\nCandidate 2: {b}")
        P2.append(dict(item=i, first=first, right="1" if first == "true" else "2", prompt=prompt))
with cf.ThreadPoolExecutor(6) as ex:
    A1 = list(ex.map(ask, [p["prompt"] for p in P1]))
    A2 = list(ex.map(ask, [p["prompt"] for p in P2]))
for p, a in zip(P1, A1):
    p["raw"] = a; p["answer"] = parse(a, "ABCD")
for p, a in zip(P2, A2):
    p["raw"] = a; p["answer"] = parse(a, "12")
errs = sum(str(p["answer"]).startswith("ERROR") for p in P1 + P2)
unparsed = sum(p["answer"] == "?" for p in P1 + P2)
print(f"model {MODEL}, default sampling, extended thinking off; {len(P1)} cloze prompts, {len(P2)} judge prompts; failed calls {errs}; replies without a choice {unparsed}")
out = dict(model=MODEL, date="2026-11-11")
print("Part 1. the constructed cloze items (100 Looking-Glass items, four choices, chance 0.25)")
acc = {o: np.mean([p["answer"] == p["right"] for p in P1 if p["order"] == o]) for o in ("shuffled", "reversed")}
both = [(P1[2 * i]["answer"] == P1[2 * i]["right"], P1[2 * i + 1]["answer"] == P1[2 * i + 1]["right"]) for i in range(len(items))]
changed = np.mean([a != b for a, b in both])
letters = {k: sum(p["answer"] == k for p in P1) for k in L}
print(f"  accuracy, shuffled order {acc['shuffled']:.2f}; reversed order {acc['reversed']:.2f}; right in both {np.mean([a and b for a, b in both]):.2f}; "
      f"right in one order only {changed:.2f}; answers by letter {letters}")
out["cloze"] = dict(acc=acc, both=float(np.mean([a and b for a, b in both])), one_only=float(changed), letters=letters)
print("Part 2. pairwise judge (60 passages; true next sentence against a sentence from elsewhere in the book)")
acc2 = {f: np.mean([p["answer"] == p["right"] for p in P2 if p["first"] == f]) for f in ("true", "other")}
pairs = [(P2[2 * i]["answer"], P2[2 * i + 1]["answer"]) for i in range(60)]
consistent = np.mean([(a == "1" and b == "2") or (a == "2" and b == "1") for a, b in pairs])
ones = np.mean([p["answer"] == "1" for p in P2])
print(f"  right when the true sentence is shown first {acc2['true']:.3f}, second {acc2['other']:.3f}; "
      f"same choice of sentence in both orders {consistent:.3f}; share of answers '1' {ones:.3f}")
out["judge"] = dict(acc=acc2, consistent=float(consistent), share_one=float(ones))
out["P1"] = [{k: v for k, v in p.items() if k != "prompt"} for p in P1]
out["P2"] = [{k: v for k, v in p.items() if k != "prompt"} for p in P2]
out["example_prompt_cloze"] = P1[0]["prompt"]; out["example_prompt_judge"] = P2[0]["prompt"]
json.dump(out, open(os.path.join(HERE, "outputs", "judge.json"), "w"), indent=0, default=float)
