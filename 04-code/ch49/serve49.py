"""Chapter 49, 49.2: a hundred-line API for the capstone, standard library only.  Usage: python3 serve49.py  (PORT 8049, MODEL fp32)
POST /v1/complete  {"prompt", "max_tokens" (1-200), "temperature" (0.8), "seed" (0), "stream" (false)}
                   stream=true sends server-sent events, one "data:" line per token, then a final line with the usage.
POST /v1/probe     {"prompts": [...]}: log-probability of the five likeliest next tokens for each prompt (the fingerprint of 49.6).
POST /admin/swap   {"model": "fp32"|"int8"} or {"tokenizer": "capstone"|"tiny2048"}: hot swap, for the lab.
GET  /health, GET /metrics.  One JSON line per request to logs/requests.jsonl (the dashboard's data).  One request at a time."""
import os, sys, json, time, traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from core49 import *
from tokenizers import Tokenizer
from tiny.tokenizer import load_bpe
PORT, KIND = int(os.environ.get("PORT", 8049)), os.environ.get("MODEL", "fp32")
LOG = open(os.path.join(HERE, "logs", os.environ.get("LOGNAME49", "requests") + ".jsonl"), "a")
class Capstone8k:
    name = "capstone"
    def __init__(s): s.t = Tokenizer.from_file(os.path.join(ART, "tokenizer.json"))
    def encode(s, text): return s.t.encode(text).ids
    def decode(s, ids): return s.t.decode(ids)
class Tiny2048:
    name = "tiny2048"
    def __init__(s): s.t = load_bpe(os.path.join(HERE, "..", "tiny", "tiny_bpe_2048.txt"))
    def encode(s, text): return list(s.t.encode(text))
    def decode(s, ids): return s.t.decode(ids)
S = dict(kind=KIND, tok=Capstone8k(), n=0, errors=0)
S["model"], S["cfg"] = load_model(KIND)
def version(): return f"{S['cfg']['name']}@{S['cfg']['step']}/{S['kind']}/{S['tok'].name}"
class H(BaseHTTPRequestHandler):
    def log_message(s, *a): pass
    def send_json(s, code, obj):
        b = json.dumps(obj).encode(); s.send_response(code); s.send_header("Content-Type", "application/json")
        s.send_header("Content-Length", str(len(b))); s.end_headers(); s.wfile.write(b)
    def do_GET(s):
        if s.path == "/health": return s.send_json(200, dict(ok=True, version=version()))
        if s.path == "/metrics": return s.send_json(200, dict(requests=S["n"], errors=S["errors"], version=version()))
        s.send_json(404, dict(error="not found"))
    def do_POST(s):
        t0 = time.perf_counter(); rec = dict(sim=s.headers.get("X-Sim-Time"), path=s.path, version=version())
        try:
            body = json.loads(s.rfile.read(int(s.headers.get("Content-Length", 0))) or b"{}")
            if s.path == "/admin/swap":
                if "model" in body: S["model"], S["cfg"] = load_model(body["model"]); S["kind"] = body["model"]
                if "tokenizer" in body: S["tok"] = Tiny2048() if body["tokenizer"] == "tiny2048" else Capstone8k()
                return s.send_json(200, dict(version=version()))
            if s.path == "/v1/probe":
                out = []
                for p in body["prompts"]:
                    lp = first_token_logprobs(S["model"], S["tok"].encode(p)); v, i = lp.topk(5)
                    out.append(dict(prompt=p, top=[[int(a), round(float(b), 6)] for a, b in zip(i, v)]))
                return s.send_json(200, dict(version=version(), probes=out))
            if s.path != "/v1/complete": return s.send_json(404, dict(error="not found"))
            prompt, mx = body.get("prompt", ""), int(body.get("max_tokens", 32))
            if not isinstance(prompt, str) or not prompt.strip() or not 1 <= mx <= 200:
                rec.update(status=400, error="bad request"); return s.send_json(400, dict(error="prompt must be text and max_tokens 1 to 200"))
            ids = S["tok"].encode(prompt); rec.update(prompt_tokens=len(ids), stream=bool(body.get("stream")))
            if len(ids) > 200:
                rec.update(status=413, error="prompt too long"); return s.send_json(413, dict(error="prompt longer than 200 tokens"))
            gen = generate(S["model"], ids, mx, float(body.get("temperature", 0.8)), int(body.get("seed", 0)), S["cfg"]["context"])
            out, times = [], []
            if body.get("stream"):
                s.send_response(200); s.send_header("Content-Type", "text/event-stream"); s.end_headers()
                for tid, t in gen:
                    out.append(tid); times.append(t); s.wfile.write(b"data: " + json.dumps(dict(token=S["tok"].decode([tid]))).encode() + b"\n\n")
                text = S["tok"].decode(out)
                s.wfile.write(b"data: " + json.dumps(dict(done=True, completion_tokens=len(out))).encode() + b"\n\n")
            else:
                for tid, t in gen: out.append(tid); times.append(t)
                text = S["tok"].decode(out)
            rec.update(status=200, completion_tokens=len(out), ttft_ms=1000 * times[0], finish="window" if len(ids) + len(out) >= S["cfg"]["context"] else "length", text=text)
            if not body.get("stream"):
                s.send_json(200, dict(text=text, model=version(), usage=dict(prompt_tokens=len(ids), completion_tokens=len(out))))
        except Exception as e:
            S["errors"] += 1; rec.update(status=500, error=f"{type(e).__name__}: {e}"[:200])
            try: s.send_json(500, dict(error="internal error"))
            except Exception: pass
        finally:
            if s.path == "/v1/complete":
                S["n"] += 1; rec["total_ms"] = 1000 * (time.perf_counter() - t0); LOG.write(json.dumps(rec) + "\n"); LOG.flush()
if __name__ == "__main__":
    print("serving", version(), "on port", PORT, flush=True); HTTPServer(("127.0.0.1", PORT), H).serve_forever()
