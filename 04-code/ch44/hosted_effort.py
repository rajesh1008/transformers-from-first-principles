"""Chapter 44, test-time compute on a hosted model: 30 multiplications of two four-digit numbers (seed 1865), asked of Claude
Haiku 4.5 (claude-haiku-4-5-20251001) through Anthropic's Messages API with extended thinking off, and on with thinking
budgets of 1,024 and 4,096 tokens.  The prompt asks for the product only.  Records exact accuracy, output tokens (thinking
included, as billed) and wall-clock seconds per request.  The key is read from ../../.env into this process only; it is never
printed or written.  Writes outputs/hosted_effort.json."""
import os, re, json, time, random, urllib.request, concurrent.futures as cf
HERE = os.path.dirname(os.path.abspath(__file__)); MODEL = os.environ.get("MODEL", "claude-haiku-4-5-20251001")
def _key():
    for l in open(os.path.join(HERE, "..", "..", ".env")):
        if l.startswith("ANTHROPIC_API_KEY="): return l.split("=", 1)[1].strip().strip('"')
KEY = _key()
def ask(q, budget):
    body = dict(model=MODEL, max_tokens=(budget + 300) if budget else 64, messages=[dict(role="user", content=q)])
    body["thinking"] = {"type": "enabled", "budget_tokens": budget} if budget else {"type": "disabled"}
    for attempt in range(4):
        try:
            t0 = time.perf_counter()
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(), method="POST",
                                         headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=300)); dt = time.perf_counter() - t0
            text = "".join(c.get("text", "") for c in r["content"] if c["type"] == "text")
            return dict(text=text, out=r["usage"]["output_tokens"], seconds=dt)
        except Exception as e:
            err = type(e).__name__; time.sleep(5 * (attempt + 1))
    return dict(text="", out=0, seconds=0, error=err)
rng = random.Random(1865); probs = [(rng.randint(1000, 9999), rng.randint(1000, 9999)) for _ in range(30)]
R = dict(model=MODEL, n=len(probs), conditions=[])
for budget in (0, 1024, 4096):
    qs = [f"What is {a} × {b}? Reply with the product only, as digits with no commas." for a, b in probs]
    with cf.ThreadPoolExecutor(6) as ex: res = list(ex.map(lambda q: ask(q, budget), qs))
    ok = [re.sub(r"[^0-9]", "", x["text"]) == str(a * b) for x, (a, b) in zip(res, probs)]
    C = dict(budget=budget, exact=sum(ok), mean_out=sum(x["out"] for x in res) / len(res), median_seconds=sorted(x["seconds"] for x in res)[len(res) // 2],
             errors=sum("error" in x for x in res), wrong_examples=[(a, b, x["text"][:40]) for x, (a, b), o in zip(res, probs, ok) if not o][:3])
    R["conditions"].append(C)
    print(f"thinking {'off' if not budget else f'budget {budget:,}'}: exact {C['exact']}/30; mean output tokens {C['mean_out']:.0f}; median seconds {C['median_seconds']:.1f}; errors {C['errors']}", flush=True)
json.dump(R, open(os.path.join(HERE, "outputs", "hosted_effort.json"), "w"), indent=1)
