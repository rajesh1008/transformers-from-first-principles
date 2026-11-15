"""Chapter 31: a small real server for GPT-2 small (tiny/load.py + tiny/serve.py), twenty simulated clients.
Static batching against continuous batching on the same twenty requests (workload.requests), wall clock, one thread.
Then the breakage: continuous batching with too few cache pages and no reservation, where a request stalls.
Prompts are cut from Chapter 22's held-out Carroll text; request 2 is the running sentence.  Greedy decoding."""
from _common import *
import torch, time, sys, numpy as np
torch.set_num_threads(1)
from tiny.load import GPT2
from tiny import serve as S
from transformers import AutoTokenizer
from workload import requests, cost_model
from sim import simulate
tk = AutoTokenizer.from_pretrained(CKPT("gpt2")); gm = GPT2.from_dir(CKPT("gpt2")); g = held_text_ids(tk)
REQS = requests()
def prompt(r):
    return tk.encode(SENTENCE) if r.get("sentence") else g[r["id"] * 97: r["id"] * 97 + r["P"]]

def run(scheme, B=16, pages=None, page=16, log=False):
    reqs = [dict(r) for r in REQS]; pend = sorted(reqs, key=lambda r: r["arrive"]); wait, run_, done = [], [], []
    kv = S.PagedKV(pages, page) if pages else None; stalls = {}
    t0 = time.perf_counter(); now = lambda: time.perf_counter() - t0
    while pend or wait or run_:
        while pend and pend[0]["arrive"] <= now(): wait.append(pend.pop(0))
        if not wait and not run_:
            time.sleep(max(0, pend[0]["arrive"] - now())); continue
        if scheme == "static":
            batch = wait[:B]; del wait[:B]
            for r in batch:
                ids = prompt(r); s, c = S.prefill(gm, ids); r.update(ids=list(ids), cache=c, s=s, first=now(), made=0)
            steps = max(r["O"] for r in batch)
            for k in range(steps):
                toks = []
                for r in batch:
                    n = int(r["s"].argmax()); r["ids"].append(n); toks.append(n)
                    if r["made"] < r["O"]:
                        r["made"] += 1
                        if r["made"] == r["O"]: r["end"] = now()
                if k == steps - 1: break
                sc = S.decode_step(gm, [r["cache"] for r in batch], toks, [len(r["ids"]) - 1 for r in batch])
                for i, r in enumerate(batch): r["s"] = sc[i]
            for r in batch: r["back"] = now(); done.append(r)       # the batch's results return together
            continue
        while wait and len(run_) < B:
            r = wait[0]
            if kv and not kv.can_alloc(r["id"], r["P"]): break
            wait.pop(0); ids = prompt(r)
            if kv: kv.alloc(r["id"], len(ids))
            s, c = S.prefill(gm, ids); r.update(ids=list(ids), cache=c, s=s, first=now(), made=1); r["ids"].append(int(s.argmax()))
            run_.append(r)
        active = [r for r in run_ if r["made"] < r["O"]]
        if kv:
            ok = []
            for r in active:
                if kv.alloc(r["id"], 1): ok.append(r)
                else: stalls[r["id"]] = stalls.get(r["id"], 0) + 1
            active = ok
            if not active and run_:
                print(f"  t {now():6.2f} s: no request can take a page; {len(run_)} running, {len(kv.free_pages)} pages free; stopping", flush=True)
                return reqs, dict(deadlock=True, stalls=stalls, done=len(done))
        if active:
            sc = S.decode_step(gm, [r["cache"] for r in active], [r["ids"][-1] for r in active], [len(r["ids"]) - 1 for r in active])
            for i, r in enumerate(active):
                r["ids"].append(int(sc[i].argmax())); r["made"] += 1
        for r in [r for r in run_ if r["made"] >= r["O"]]:
            run_.remove(r); r["end"] = r["back"] = now(); done.append(r)
            if kv: kv.free(r["id"])
    return reqs, dict(stalls=stalls, done=len(done), makespan=now())

def report(name, reqs, info):
    ttft = np.array([r["first"] - r["arrive"] for r in reqs]); back = np.array([r["back"] - r["arrive"] for r in reqs])
    toks = sum(r["O"] for r in reqs); span = max(r["back"] for r in reqs) - min(r["arrive"] for r in reqs)
    s = [r for r in reqs if r.get("sentence")][0]
    print(f"  {name:10s}: {toks} tokens in {span:5.1f} s = {toks / span:5.1f} tokens/s; first token mean {ttft.mean():5.2f} s, "
          f"p50 {np.median(ttft):5.2f}, max {ttft.max():5.2f}; reply back mean {back.mean():5.2f} s, p50 {np.median(back):5.2f}, max {back.max():5.2f}")
    print(f"  {'':10s}  the running sentence (6 prompt tokens, 16 new, arrives at {s['arrive']:.2f} s): first token after {s['first'] - s['arrive']:.2f} s, "
          f"reply back after {s['back'] - s['arrive']:.2f} s; text {tk.decode(s['ids'][6:6 + 16])!r}")
    return dict(tok_s=toks / span, ttft_mean=float(ttft.mean()), ttft_max=float(ttft.max()), back_mean=float(back.mean()),
                back_max=float(back.max()), sentence_ttft=s["first"] - s["arrive"], sentence_back=s["back"] - s["arrive"])

print("The twenty requests (arrival s, prompt tokens, new tokens):")
print("  " + "; ".join(f"{r['id']}: {r['arrive']:.1f}/{r['P']}/{r['O']}" for r in REQS))
print(f"  total prompt tokens {sum(r['P'] for r in REQS)}, total new tokens {sum(r['O'] for r in REQS)}")
out = {}
print("\nReal runs on GPT-2 small, B up to 16 (wall clock):")
for sch in ("static", "continuous"):
    reqs, info = run(sch); out[sch] = report(sch, reqs, info)
dec, pre = cost_model()
print("\nThe simulator on the same requests (measured step-cost model):")
for sch in ("static", "continuous"):
    m = simulate(REQS, dec, pre, sch, B=16)
    print(f"  {sch:10s}: {m['tok_s']:5.1f} tokens/s; first token mean {m['ttft_mean']:5.2f} s; reply mean {m['lat_mean']:5.2f} s")
    out["sim_" + sch] = {k: float(v) for k, v in m.items()}
print("\nBreakage: continuous batching, 24 pages of 16 slots (384 cache slots), admission checks only the prompt")
reqs, info = run("continuous", pages=24, page=16)
print(f"  finished {info['done']} of 20; stalled steps per request {info['stalls']}")
out["break"] = dict(done=info["done"], stalls={str(k): v for k, v in info["stalls"].items()}, deadlock=info.get("deadlock", False))
save_json("serve_real", out)
