"""Chapter 49, 49.5: a simulated first week of traffic, driven through the real API.  Usage: python3 traffic49.py [N=10000]
Starts serve49.py, then sends N requests in order.  Only the clock is simulated: each request carries an X-Sim-Time header
(seconds from Monday 00:00) drawn from a seeded arrival process with a daily cycle; latency, tokens and errors are measured.
Prompts: the running sentence first; then openings of 3 to 10 words from sentences of The Time Machine (held out in Chapter 48),
2 per cent long prompts (about 150 tokens), 0.5 per cent empty and 0.3 per cent too long.  30 per cent stream.
Scheduled changes, as a team might make them in a first week (49.5):
  Wednesday 12:00  the model is swapped from 32-bit to the 8-bit artefact (a quiet change: same version name apart from the suffix)
  Friday 09:00     the tokenizer is hot-swapped for Tiny's 2,048-token BPE (the breakage of Lab 49)
  Friday 15:00     the tokenizer is swapped back (the rollback)
Every simulated hour, /v1/probe is called with eight fixed prompts and the answers go to outputs/probes49.jsonl (49.6).
-> logs/requests.jsonl (from the server), outputs/traffic49.json, outputs/probes49.jsonl.  Seed 1865.
Added 16 November 2026 for exercises 49.7 and 49.8: TAG=name writes logs/requests_name.jsonl, outputs/traffic49_name.json and
outputs/probes49_name.jsonl instead (the chapter's files are left alone); GUARD is passed to the server; and the client counts a
streamed 200 whose body has no final {"done": true} event as a failure (client_view in the output)."""
import os, sys, json, time, re, subprocess, urllib.request, urllib.error
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
PORT = int(os.environ.get("PORT", 8049)); URL = f"http://127.0.0.1:{PORT}"
TAG = os.environ.get("TAG", ""); SFX = f"_{TAG}" if TAG else ""
rng = np.random.default_rng(1865); DAY = 86400
for f in (f"logs/requests{SFX}.jsonl", f"outputs/probes49{SFX}.jsonl"):
    p = os.path.join(HERE, f)
    if os.path.exists(p): os.remove(p)
srv = subprocess.Popen([sys.executable, os.path.join(HERE, "serve49.py")], cwd=HERE, env=dict(os.environ, PORT=str(PORT), MODEL="fp32", LOGNAME49=f"requests{SFX}"))
for _ in range(120):
    try: urllib.request.urlopen(URL + "/health", timeout=2); break
    except Exception: time.sleep(0.5)
def post(path, obj, sim=None):
    rq = urllib.request.Request(URL + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json", "X-Sim-Time": str(sim)})
    try:
        with urllib.request.urlopen(rq, timeout=120) as r: return r.status, r.read().decode()
    except urllib.error.HTTPError as e: return e.code, e.read().decode()
text = open(os.path.join(HERE, "..", "ch48", "corpus", "heldout_timemachine.txt"), encoding="utf-8").read()
sents = [s.split() for s in re.split(r"(?<=[.?!])\s+", " ".join(text.split())) if len(s.split()) >= 12]
long_src = " ".join(text.split()[2000:2120])
# Arrival times: a daily cycle (quiet at night, busiest mid-afternoon), more traffic as the week goes on.
w = np.array([(0.25 + np.sin(np.pi * ((t % DAY) / DAY - 0.25)) ** 4) * (1 + 0.08 * (t // DAY)) for t in range(0, 7 * DAY, 60)])
sim = np.sort(rng.choice(np.arange(0, 7 * DAY, 60), size=N, p=w / w.sum()) + rng.integers(0, 60, N))
EVENTS = [(2 * DAY + 12 * 3600, dict(model="int8")), (4 * DAY + 9 * 3600, dict(tokenizer="tiny2048")), (4 * DAY + 15 * 3600, dict(tokenizer="capstone"))]
PROBES = ["the cat sat on the mat.", "The Time Traveller", "It was a", "\u201cI do not", "She looked at", "In the morning", "Mr. Darcy", "The door opened and"]
ev_i, next_probe, done, codes = 0, 0, [], {}; client = dict(ok=0, failed=0, stream_ok=0, stream_incomplete=0)
t_start = time.time()
for i, t in enumerate(sim):
    t = int(t)
    while next_probe <= t:
        c, b = post("/v1/probe", dict(prompts=PROBES), next_probe)
        open(os.path.join(HERE, "outputs", f"probes49{SFX}.jsonl"), "a").write(json.dumps(dict(sim=next_probe, **json.loads(b))) + "\n"); next_probe += 3600
    while ev_i < len(EVENTS) and EVENTS[ev_i][0] <= t:
        c, b = post("/admin/swap", EVENTS[ev_i][1], EVENTS[ev_i][0]); done.append(dict(sim=EVENTS[ev_i][0], change=EVENTS[ev_i][1], answer=json.loads(b))); ev_i += 1
    u = rng.random()
    if i == 0: req = dict(prompt="the cat sat on the mat", max_tokens=32, temperature=0.8, seed=1)
    elif u < 0.005: req = dict(prompt="", max_tokens=32)
    elif u < 0.008: req = dict(prompt=" ".join(text.split()[5000:5400]), max_tokens=16)
    elif u < 0.028: req = dict(prompt=long_src, max_tokens=int(rng.integers(8, 49)))
    else:
        s = sents[int(rng.integers(len(sents)))]; req = dict(prompt=" ".join(s[:int(rng.integers(3, 11))]), max_tokens=int(rng.integers(8, 49)))
    req.update(temperature=req.get("temperature", 0.8), seed=req.get("seed", int(rng.integers(1 << 30))), stream=bool(rng.random() < 0.3) and i > 0)
    c, b = post("/v1/complete", req, t); codes[c] = codes.get(c, 0) + 1
    if c == 200 and req["stream"]:
        good = '"done": true' in b; client["stream_ok" if good else "stream_incomplete"] += 1; client["ok" if good else "failed"] += 1
    else: client["ok" if c == 200 else "failed"] += 1
    if i == 0: first = dict(request=req, status=c, answer=b)
    if (i + 1) % 1000 == 0: print(f"{i + 1} requests, {time.time() - t_start:.0f} s, codes {codes}", flush=True)
srv.terminate()
json.dump(dict(n=N, codes={str(k): v for k, v in codes.items()}, client_view=client, events=done, first=first, wall_seconds=time.time() - t_start), open(os.path.join(HERE, "outputs", f"traffic49{SFX}.json"), "w"), indent=1)
print("done", codes, client)
