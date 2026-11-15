"""Chapter 26, section 26.8: the same request sent to a hosted model again and again (one run, 11 November 2026).
Model claude-haiku-5-5 (Anthropic Messages API), extended thinking off, at most 40 output tokens, default sampling.
First asks once with temperature 0 to record whether the API accepts it (Chapter 25 found it refused for this model).
Then three prompts, each sent 20 times, one after another:
  P1  'Reply with one word: the word that most often follows "the cat sat on the".'
  P2  a cloze item from Chapter 25 (outputs/cloze_items.json, Looking-Glass item 0) with four lettered choices
  P3  'Write one short sentence about a cat sitting on a mat.'
Counts distinct replies per prompt.  The key is read from ../../.env into this process only; never printed or written.
Writes outputs/repeat_hosted.json (prompts, replies; no key)."""
import _common
import os, json, time, urllib.request, urllib.error
from _common import HERE
MODEL = "claude-haiku-5-5"
def _key():
    for l in open(os.path.join(HERE, "..", "..", ".env")):
        if l.startswith("ANTHROPIC_API_KEY="):
            return l.split("=", 1)[1].strip().strip('"')
KEY = _key()
def ask(prompt, **extra):
    body = dict(model=MODEL, max_tokens=40, thinking={"type": "disabled"}, messages=[dict(role="user", content=prompt)]); body.update(extra)
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(), method="POST",
                                 headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    for attempt in range(4):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=60))
            return "".join(b.get("text", "") for b in r["content"]).strip()
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="ignore")[:300]
            if e.code in (400, 404):
                return f"ERROR {e.code} {msg}"
            time.sleep(3 + 3 * attempt)
        except Exception as e:
            time.sleep(3 + 3 * attempt); msg = type(e).__name__
    return "ERROR " + msg
it = [i for i in json.load(open(os.path.join(HERE, "..", "ch25", "outputs", "cloze_items.json"))) if i["src"] == "glass"][0]
opts = [it["word"]] + it["distractors"]
P = {"P1": 'Reply with one word: the word that most often follows "the cat sat on the".',
     "P2": ("The passage below stops just before a word. Which of the four words comes next in the original text? "
            "Reply with the letter only.\n\nPassage: ..." + it["context_tail"][-400:] + "\n\n" + "\n".join(f"{l}. {w}" for l, w in zip("ABCD", opts))),
     "P3": "Write one short sentence about a cat sitting on a mat."}
O = dict(model=MODEL, date=time.strftime("%Y-%m-%d"), temperature0=ask(P["P1"], temperature=0.0), prompts=P, replies={})
print("temperature 0 accepted?", "no: " + O["temperature0"][:160] if O["temperature0"].startswith("ERROR") else "yes, reply " + repr(O["temperature0"]))
for k, p in P.items():
    O["replies"][k] = [ask(p) for _ in range(20)]
    reps = O["replies"][k]
    from collections import Counter
    c = Counter(reps)
    print(f"{k}: {len(c)} distinct replies in 20; most common {c.most_common(3)}")
json.dump(O, open(os.path.join(HERE, "outputs", "repeat_hosted.json"), "w"), indent=1)
