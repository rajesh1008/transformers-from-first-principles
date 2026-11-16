"""Exercise 49.7 check (16 November 2026): start serve49.py with GUARD=1 and send the week's three scheduled changes
in order, with a probe after each, instead of replaying all 10,000 requests. Writes outputs/ex49_guard.json."""
import json, os, subprocess, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); PORT = int(os.environ.get("PORT", 8054)); URL = f"http://127.0.0.1:{PORT}"
srv = subprocess.Popen([sys.executable, os.path.join(HERE, "serve49.py")], cwd=HERE, env=dict(os.environ, PORT=str(PORT), MODEL="fp32", GUARD="1", LOGNAME49="requests_ex49guard"))
def post(path, body):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=600) as f: return f.status, json.loads(f.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read())
for _ in range(600):
    try: urllib.request.urlopen(URL + "/health", timeout=5); break
    except Exception: time.sleep(1)
out = []
try:
    for name, ch in [("Wednesday 12:00", dict(model="int8")), ("Friday 09:00", dict(tokenizer="tiny2048")), ("Friday 15:00", dict(tokenizer="capstone"))]:
        st, ans = post("/admin/swap", ch); pst, pr = post("/v1/probe", dict(prompts=["The Time Traveller"]))
        out.append(dict(when=name, change=ch, status=st, answer=ans, probe_version=pr.get("version"), probe_top=pr["probes"][0]["top"][:3]))
        print(f"49.7 {name} {ch}: status {st}, version after {pr.get('version')}", flush=True)
finally:
    srv.terminate(); srv.wait()
json.dump(out, open(os.path.join(HERE, "outputs", "ex49_guard.json"), "w"), indent=1)
