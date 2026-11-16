"""Chapter 49 exercises 49.8 (from the chapter's own week), 49.9, 49.11 and 49.12 (added 16 November 2026).  CPU, one thread, seed 1865.
  49.8  the client's view of the chapter's week, from logs/requests.jsonl: the server sends its final {"done": true} event only after
        the last token, so every logged stream with status 500 is a 200 the client received without that event.
  49.9  a probe alert on the change of any of the 40 probe log-probabilities from the previous hourly probe (outputs/probes49.jsonl);
        the thresholds at which the 8-bit swap raises exactly one alert, and what else alerts at those thresholds.
  49.11 a 4-bit artefact (levels -7 to 7, one 32-bit scale per group of 32 numbers along each row; stored in int8 containers, so the
        file is not smaller than the 8-bit one) written to artefact/capstone-int4.safetensors (not in the manifest), served by
        serve49.py (MODEL=int4) for one /v1/probe call, and compared with fp32 and int8: the largest probe change against fp32 and the
        share of known words in 200 completions of 32 tokens (temperature 0.8) from seeded openings of The Time Machine.
  49.12 a redacted log: counts and timings for every request, text for one request in a hundred chosen by a seeded hash; the
        dashboard's quality panel (share of known words per day) from the redacted log against the full log.
-> outputs/ex49.json"""
import os, sys, json, re, hashlib, subprocess, time, urllib.request
import numpy as np
from core49 import *
from safetensors.torch import save_file, load_file
from tokenizers import Tokenizer
O = os.path.join(HERE, "outputs"); DAY = 86400; R = {}
LOG = [json.loads(l) for l in open(os.path.join(HERE, "logs", "requests.jsonl"))]
st = [r for r in LOG if r.get("stream")]; hidden = sum(r["status"] == 500 for r in st)
client200 = sum(r["status"] == 200 for r in LOG) + hidden
R["ex8"] = dict(requests=len(LOG), streams=len(st), streams_500_after_200=hidden, client_saw_200=client200,
                client_counting_done=dict(ok=client200 - hidden, failed=len(LOG) - (client200 - hidden)))
print(f"49.8 week of {len(LOG):,}: {len(st):,} streams, {hidden} of them failed after status 200; the client saw 200 on {client200:,};"
      f" counting a stream without the final event as a failure, {client200 - hidden:,} succeeded and {len(LOG) - client200 + hidden:,} failed")
P = [json.loads(l) for l in open(os.path.join(O, "probes49.jsonl"))]
def change(a, b):
    m = 0.0
    for pa, pb in zip(a["probes"], b["probes"]):
        da, db = dict(map(tuple, pa["top"])), dict(map(tuple, pb["top"]))
        for k in set(da) | set(db):
            if k in da and k in db: m = max(m, abs(da[k] - db[k]))
            else: m = max(m, float("inf"))      # a token entered or left the top five
    return m
ch = [dict(sim=P[i]["sim"], version=P[i]["version"], prev=P[i - 1]["version"], change=change(P[i - 1], P[i])) for i in range(1, len(P))]
swap = [c for c in ch if c["prev"].endswith("/fp32/capstone") and c["version"].endswith("/int8/capstone")]
steady = [c["change"] for c in ch if c["prev"] == c["version"]]
lo = max(steady) if steady else 0.0; hi = swap[0]["change"] if swap else None
others = [c for c in ch if c["prev"] != c["version"] and c is not swap[0]]
R["ex9"] = dict(hours=len(ch), int8_swap_change=hi, largest_steady_change=lo, threshold_range=[lo, hi],
                other_version_changes=[dict(sim=c["sim"], frm=c["prev"], to=c["version"], change=c["change"] if np.isfinite(c["change"]) else "top five changed") for c in others])
print(f"49.9 {len(ch)} hourly comparisons; the 8-bit swap moved a probe by {hi:.6f}; the largest change between probes of the same version was {lo:.6f};"
      f" any threshold from {lo:.6f} up to (not including) {hi:.6f} alerts once for the swap; other version changes: " +
      "; ".join(f"{c['prev'].split('/', 1)[1]} -> {c['version'].split('/', 1)[1]} at hour {c['sim'] // 3600}: {'top five changed' if not np.isfinite(c['change']) else round(c['change'], 6)}" for c in others))
# 49.11
t32 = load_file(os.path.join(ART, "capstone-fp32.safetensors")); q4 = {}
for n, p in t32.items():
    if p.dim() == 2:
        r, c = p.shape; g = p.reshape(r, c // 32, 32); s = g.abs().amax(-1, keepdim=True) / 7.0; s[s == 0] = 1.0
        q4[n + ".q"] = torch.clamp(torch.round(g / s), -7, 7).to(torch.int8).reshape(r, c).contiguous(); q4[n + ".scale"] = s.reshape(r, c // 32).contiguous()
    else: q4[n] = p
save_file(q4, os.path.join(ART, "capstone-int4.safetensors"), metadata=dict(format="pt", quant="4-bit levels -7..7, one scale per 32 numbers, int8 containers"))
tk = Tokenizer.from_file(os.path.join(ART, "tokenizer.json"))
PROBES = ["the cat sat on the mat.", "The Time Traveller", "It was a", "\u201cI do not", "She looked at", "In the morning", "Mr. Darcy", "The door opened and"]
models = {k: load_model(k)[0] for k in ("fp32", "int8", "int4")}
lp = {k: [first_token_logprobs(m, tk.encode(p).ids) for p in PROBES] for k, m in models.items()}
text = open(os.path.join(HERE, "..", "ch48", "corpus", "heldout_timemachine.txt"), encoding="utf-8").read()
sents = [s.split() for s in re.split(r"(?<=[.?!])\s+", " ".join(text.split())) if len(s.split()) >= 12]
W = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")
known = {w.lower() for w in W.findall(open(os.path.join(HERE, "..", "ch48", "corpus", "train.txt"), encoding="utf-8").read())}
rng = np.random.default_rng(1865); jobs = [(" ".join(sents[int(rng.integers(len(sents)))][:int(rng.integers(3, 11))]), int(rng.integers(1 << 30))) for _ in range(200)]
R["ex11"] = {}
for k, m in models.items():
    top5 = [set(map(int, l.topk(5).indices)) for l in lp["fp32"]]
    pc = max(float((lp[k][i][list(top5[i])] - lp["fp32"][i][list(top5[i])]).abs().max()) for i in range(len(PROBES)))
    tops = sum(int(lp[k][i].argmax()) == int(lp["fp32"][i].argmax()) for i in range(len(PROBES)))
    words = []
    for prompt, seed in jobs: words += W.findall(tk.decode([t for t, _ in generate(m, tk.encode(prompt).ids, 32, 0.8, seed)]))
    ks = sum(w.lower() in known for w in words) / len(words)
    R["ex11"][k] = dict(probe_change_vs_fp32=pc, same_top_token=tops, known_share=ks, words=len(words), file_bytes=os.path.getsize(os.path.join(ART, f"capstone-{k}.safetensors")))
    print(f"49.11 {k}: largest probe change against fp32 {pc:.4f}; top token unchanged on {tops} of 8 probes; known words {100 * ks:.1f}% of {len(words):,}", flush=True)
srv = subprocess.Popen([sys.executable, os.path.join(HERE, "serve49.py")], cwd=HERE, env=dict(os.environ, PORT="8053", MODEL="int4", LOGNAME49="requests_ex11"))
for _ in range(240):
    try: urllib.request.urlopen("http://127.0.0.1:8053/health", timeout=2); break
    except Exception: time.sleep(0.5)
rq = urllib.request.Request("http://127.0.0.1:8053/v1/probe", data=json.dumps(dict(prompts=PROBES)).encode(), headers={"Content-Type": "application/json"})
served = json.loads(urllib.request.urlopen(rq, timeout=120).read()); srv.terminate()
direct = [[[int(a), round(float(b), 6)] for a, b in zip(l.topk(5).indices, l.topk(5).values)] for l in lp["int4"]]
R["ex11"]["served_version"] = served["version"]; R["ex11"]["served_matches_direct"] = all(p["top"] == d for p, d in zip(served["probes"], direct))
print(f"49.11 served as {served['version']}; the server's probe answers match the direct computation: {R['ex11']['served_matches_direct']}")
# 49.12
keep = lambda i: int(hashlib.sha256(f"1865:{i}".encode()).hexdigest(), 16) % 100 == 0
red = [{k: v for k, v in r.items() if k != "text" or keep(i)} for i, r in enumerate(LOG)]
def ks_day(rs, d):
    ws = [w for r in rs if d * DAY <= int(r["sim"]) < (d + 1) * DAY and r["status"] == 200 and "text" in r for w in W.findall(r["text"])]
    return (sum(w.lower() in known for w in ws) / len(ws), len(ws), sum(1 for r in rs if d * DAY <= int(r["sim"]) < (d + 1) * DAY and "text" in r and r["status"] == 200)) if ws else (None, 0, 0)
names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]; R["ex12"] = dict(kept=sum("text" in r for r in red), days={})
for d in range(7):
    f, r_ = ks_day(LOG, d), ks_day(red, d); R["ex12"]["days"][names[d]] = dict(full=f[0], redacted=r_[0], redacted_texts=r_[2])
    print(f"49.12 {names[d]}: known words {100 * f[0]:.1f}% from every text, {100 * r_[0]:.1f}% from the {r_[2]} kept texts" if r_[0] is not None else f"49.12 {names[d]}: no kept text")
b_full = sum(len(json.dumps(r)) + 1 for r in LOG); b_red = sum(len(json.dumps(r)) + 1 for r in red)
R["ex12"]["bytes"] = dict(full=b_full, redacted=b_red); print(f"49.12 kept text for {R['ex12']['kept']} of {len(LOG):,} requests; log {b_full:,} -> {b_red:,} bytes")
json.dump(R, open(os.path.join(O, "ex49.json"), "w"), indent=1, default=str)
