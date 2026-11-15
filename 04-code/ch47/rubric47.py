"""Chapter 47, scoring the whiteboard exercise.  Six synthetic candidate answers (written for this lab, not from any real
candidate) to the whiteboard question W, each written to a target level, are scored three ways:
  K  a keyword rubric: distinct technical terms found, two terms per point, capped at 4 (the breakage: it rewards jargon);
  A  the five-anchor rubric below, applied by a hosted grader (Claude Haiku 4.5, temperature 1.0), 5 repeats per answer;
  J  a vocabulary rubric ("command of modern transformer terminology"), same grader, 5 repeats per answer.
The key is read from ../../.env into this process only; it is never printed or written.  Writes outputs/rubric47.json."""
import os, re, json, time, urllib.request, concurrent.futures as cf, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); MODEL = "claude-haiku-4-5-20251001"
def _key():
    for l in open(os.path.join(HERE, "..", "..", ".env")):
        if l.startswith("ANTHROPIC_API_KEY="): return l.split("=", 1)[1].strip().strip('"')
KEY = _key()
W = ("Tiny has 12 layers of width 128, 4 attention heads per layer, learned position rows and a 2,048-token vocabulary. "
     "Take the sentence 'the cat sat on the mat'. Walk me through what one attention head in layer 0 computes for the last token, "
     "and tell me how many bytes per token Tiny must keep, in 32-bit numbers, to generate the next token without recomputing the past.")
ANCHORS = {
 0: "No mechanism: describes behaviour or repeats terms; no account of how shares are computed or what is mixed.",
 1: "Names queries, keys and values, but cannot say how the shares are computed or what they weight.",
 2: "Correct pipeline (three projections; dot products of the last query with the keys; softmax; weighted sum of values) but misses at least two of: the scale by the square root of the head width (32), the causal mask, the head shapes, the output table; or caches the wrong thing.",
 3: "Correct pipeline including the scale, the mask and the shapes, and caches keys and values per layer, but the byte count is wrong or missing a factor.",
 4: "Everything in 3, plus the correct cache arithmetic, 2 x 12 x 128 x 4 = 12,288 bytes per token, and a check that would catch a mistake (for example shares sum to 1, or cached and uncached runs agree).",
}
ANSWERS = {
 "A": (4, "Seven tokens; the last is 'at', because mat splits into ' m' and 'at'. Its row, token row plus position row, goes through the layer norm and is multiplied by three 128 by 128 tables to give a query, a key and a value, each cut into four pieces of 32, one per head. For head 0 I dot the last token's 32-number query with the keys of all seven positions, itself included and nothing later, divide by the square root of 32, and softmax the seven scores into shares that sum to one. The head's output is the share-weighted sum of the seven values. The four heads are joined back to 128 numbers, multiplied by the output table and added to the residual. To generate quickly I keep every token's key and value in every layer: 2 times 12 layers times 128 numbers times 4 bytes, 12,288 bytes per token. I would check that each row of shares sums to one and that a cached and an uncached run pick the same next token."),
 "B": (0, "This is classic scaled dot-product self-attention from Attention Is All You Need. Multi-head attention lets the model attend to different representation subspaces, so head 0 captures syntactic dependencies while other heads capture semantics. With RoPE and GQA, plus FlashAttention for IO-awareness, the KV cache shrinks dramatically, and with PagedAttention and continuous batching you get very high throughput. The softmax normalises the attention, and the model learns that mat attends to cat because they rhyme. In production you would use MQA or MLA to compress the KV cache and quantise to FP8. Memory per token is basically the hidden size, so 128 numbers."),
 "C": (2, "Each token gets a query, a key and a value from three learned tables. For the last token you compare its query with every key using a dot product, turn the scores into weights with a softmax, and take a weighted average of the values. That average is what the head outputs. To make generation faster you cache things, I think the attention scores, so you do not have to redo them."),
 "D": (3, "The last token's row is normalised and projected by the query, key and value tables, 128 by 128 each, then split into four heads of 32. For head 0, the last query is dotted with the keys of the seven positions, earlier ones and itself only because of the causal mask, scaled by one over the square root of 32, and softmaxed. The output is the weighted sum of the seven values; the heads are concatenated and go through the output table. For the cache you keep the keys and values for each layer: 12 layers times 128 numbers times 4 bytes, 6,144 bytes per token."),
 "E": (1, "Attention lets each word look at the other words and decide which ones are important, like how a person focuses on the key words in a sentence. The head for mat probably focuses on cat and sat. There are queries, keys and values involved in deciding this. The cache stores previous results so the model is faster."),
 "F": (4, "Standard scaled dot-product attention, multi-head, no GQA in Tiny, so four key-value heads. The last token is 'at', since mat is ' m' plus 'at'. Normalise its row, project with W_q, W_k and W_v, 128 by 128, split into four heads of 32. Head 0: the last query against all seven keys, causal so nothing later, divided by the square root of 32, softmax to shares summing to one, weighted sum of the values; concatenate the heads and apply W_o. With FlashAttention the table is never materialised, but the arithmetic is the same. KV cache: 2 times 12 times 128 times 4, 12,288 bytes per token in FP32. My check would be cached against uncached logits."),
}
TERMS = ["scaled dot-product", "self-attention", "multi-head", "subspace", "rope", "gqa", "mqa", "mla", "flashattention", "pagedattention",
         "continuous batching", "kv cache", "fp8", "fp32", "io-aware", "throughput", "softmax", "causal", "w_q", "w_o"]
def keyword(a):
    low = a.lower(); found = [t for t in TERMS if t in low]; return min(4, len(found) // 2), found
RUB_A = "Score the answer with this rubric. Anchors:\n" + "\n".join(f"{k}: {v}" for k, v in ANCHORS.items())
RUB_J = "Score from 0 to 4 how well the answer demonstrates command of modern transformer terminology and state-of-the-art techniques (4 = excellent command, 0 = none)."
def ask(rubric, ans):
    prompt = (f"You are scoring a candidate's spoken answer in a machine-learning engineering interview.\n\nQuestion: {W}\n\nCandidate's answer: {ans}\n\n"
              f"{rubric}\n\nReply with a single integer from 0 to 4 and nothing else.")
    body = dict(model=MODEL, max_tokens=5, temperature=1.0, messages=[dict(role="user", content=prompt)])
    for attempt in range(4):
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(), method="POST",
                                         headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=120)); txt = "".join(c.get("text", "") for c in r["content"] if c["type"] == "text")
            m = re.search(r"[0-4]", txt); return int(m.group()) if m else None
        except Exception as e:
            time.sleep(5 * (attempt + 1))
    return None
jobs = [(rk, name) for rk in ("A", "J") for name in ANSWERS for _ in range(5)]
with cf.ThreadPoolExecutor(6) as ex: res = list(ex.map(lambda j: ask(RUB_A if j[0] == "A" else RUB_J, ANSWERS[j[1]][1]), jobs))
R = dict(model=MODEL, question=W, anchors=ANCHORS, answers={k: dict(target=v[0], text=v[1]) for k, v in ANSWERS.items()}, scores={})
print(f"grader {MODEL}, temperature 1.0, 5 repeats per answer per rubric; answers are synthetic fixtures written to a target level")
print("answer  target  words  keyword(K)  terms   anchored(A) repeats    mean   vocabulary(J) repeats  mean")
for name, (target, text) in ANSWERS.items():
    k, found = keyword(text)
    a = [s for (rk, n), s in zip(jobs, res) if rk == "A" and n == name]; j = [s for (rk, n), s in zip(jobs, res) if rk == "J" and n == name]
    R["scores"][name] = dict(target=target, words=len(text.split()), keyword=k, terms=found, anchored=a, vocab=j)
    fa = [x for x in a if x is not None]; fj = [x for x in j if x is not None]
    print(f"  {name}       {target}      {len(text.split()):3d}      {k}        {len(found):2d}     {a}   {st.mean(fa):.1f}     {j}   {st.mean(fj):.1f}")
def spearman(x, y):
    import numpy as np
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    def avg_rank(v):
        v = np.asarray(v, float); order = v.argsort(); r = np.empty(len(v)); r[order] = np.arange(len(v))
        for u in np.unique(v): r[v == u] = r[v == u].mean()
        return r
    rx, ry = avg_rank(x), avg_rank(y); return float(np.corrcoef(rx, ry)[0, 1])
tg = [v["target"] for v in R["scores"].values()]
for lab, key in (("keyword", "keyword"), ("anchored mean", "anchored"), ("vocabulary mean", "vocab")):
    vals = [v[key] if key == "keyword" else st.mean([x for x in v[key] if x is not None]) for v in R["scores"].values()]
    R[f"spearman_{key}"] = spearman(tg, vals); print(f"Spearman with the target levels, {lab}: {R[f'spearman_{key}']:.3f}")
agree = lambda key: st.mean([max(v[key].count(s) for s in set(v[key])) / len(v[key]) for v in R["scores"].values()])
R["agree_anchored"], R["agree_vocab"] = agree("anchored"), agree("vocab")
print(f"share of repeats equal to the answer's commonest score: anchored {R['agree_anchored']:.3f}, vocabulary {R['agree_vocab']:.3f}")
print(f"failed calls: {sum(r is None for r in res)} of {len(res)}")
json.dump(R, open(os.path.join(HERE, "outputs", "rubric47.json"), "w"), indent=1)
