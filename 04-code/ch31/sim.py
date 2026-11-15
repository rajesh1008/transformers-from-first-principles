"""Chapter 31: an event-driven serving simulator driven by the measured GPT-2 step costs (workload.cost_model).
Schemes: 'static' (take up to B waiting requests, prefill them padded to the longest prompt, decode padded until the longest
reply ends, no admissions meanwhile), 'continuous' (iteration-level: admit between steps, whole-prompt prefill stalls the
decode step), 'chunked' (continuous, prefill split into chunks of `chunk` tokens run with each decode step).
Queue order 'fcfs' or 'short' (shortest reply first, oracle lengths).  Memory: `pages` x `page` cache slots, either reserved
contiguously at admission for P + max_new ('contig') or taken page by page ('paged'); paged with overcommit can run out
mid-reply, and then the youngest request is pre-empted and its cache recomputed later (vLLM's recompute policy, S388)."""
import math, numpy as np

def simulate(reqs, dec, pre, scheme="continuous", B=16, order="fcfs", chunk=64, mem=None, page=16, alloc="paged", max_new=128):
    reqs = [dict(r) for r in reqs]; t = 0.0; waiting = []; running = []; done = []; pending = sorted(reqs, key=lambda r: r["arrive"])
    pad_slots = used_slots = 0; preempt = 0; stall_time = 0.0; peak_reserved = []
    def arrive(now):
        while pending and pending[0]["arrive"] <= now:
            waiting.append(pending.pop(0))
    def key(r): return (r["O"] if order == "short" else 0, r["arrive"])
    def reserved(rs):
        if mem is None: return 0
        if alloc == "contig": return sum(math.ceil((r["P"] + max_new) / page) for r in rs)
        return sum(math.ceil((r["P"] + r.get("made", 0)) / page) for r in rs)
    while pending or waiting or running:
        arrive(t)
        if not waiting and not running:
            t = pending[0]["arrive"]; continue
        waiting.sort(key=key)
        if scheme == "static":
            batch = []
            while waiting and len(batch) < B and (mem is None or reserved(batch + [waiting[0]]) <= mem): batch.append(waiting.pop(0))
            Pm = max(r["P"] for r in batch); Om = max(r["O"] for r in batch)
            t += len(batch) * pre(Pm)
            for r in batch: r["first"] = t; r["made"] = 1; r["gaps"] = []
            pad_slots += len(batch) * Pm; used_slots += sum(r["P"] for r in batch)
            for s in range(1, Om):
                dt = dec(len(batch), len(batch) * (Pm + s)); t += dt
                pad_slots += len(batch); used_slots += sum(1 for r in batch if r["made"] < r["O"])
                for r in batch:
                    if r["made"] < r["O"]:
                        r["made"] += 1; r["gaps"].append(dt)
                        if r["made"] == r["O"]: r["end"] = t
            for r in batch:
                r.setdefault("end", t); done.append(r)
            continue
        # continuous / chunked: admit
        while waiting and len(running) < B:
            r = waiting[0]
            if mem is not None:
                need = math.ceil((r["P"] + max_new) / page) if alloc == "contig" else math.ceil((r["P"] + 1) / page)
                if reserved(running) + need > mem: break
            waiting.pop(0); r.update(done_pre=0, made=0, gaps=[], last=None); running.append(r)
        dt = 0.0
        pre_list = [r for r in running if r["done_pre"] < r["P"]]
        if scheme == "continuous":
            for r in pre_list:
                dt += pre(r["P"]); r["done_pre"] = r["P"]
        else:
            budget = chunk
            for r in pre_list:
                c = min(budget, r["P"] - r["done_pre"])
                if c <= 0: break
                dt += pre(c, r["done_pre"]); r["done_pre"] += c; budget -= c
        dec_list = [r for r in running if r["done_pre"] >= r["P"] and r not in pre_list]
        dt += dec(len(dec_list), sum(r["P"] + r["made"] for r in dec_list))
        if mem is not None and alloc == "paged":
            while reserved(running) > mem and len(running) > 1:       # out of pages: pre-empt the youngest
                v = max(running, key=lambda r: r["arrive"]); running.remove(v); preempt += 1
                v.update(done_pre=0, P=v["P"] + v["made"], O=v["O"] - v["made"], made=0); v.pop("first", None) if False else None
                waiting.insert(0, v)
        t += dt
        for r in pre_list:
            if r["done_pre"] >= r["P"] and r in running:
                r["made"] += 1; r.setdefault("first", t); r["last"] = t
        for r in dec_list:
            if r in running:
                r["made"] += 1; r["gaps"].append(t - r["last"]) if r["last"] is not None else None; r["last"] = t
        for r in [r for r in running if r["made"] >= r["O"]]:
            running.remove(r); r["end"] = t; done.append(r)
        if mem is not None: peak_reserved.append((reserved(running), sum(r["P"] + r["made"] for r in running)))
    ttft = np.array([r["first"] - r["arrive"] for r in done]); lat = np.array([r["end"] - r["arrive"] for r in done])
    gaps = np.concatenate([np.array(r["gaps"]) for r in done if r["gaps"]]) if any(r["gaps"] for r in done) else np.zeros(1)
    toks = sum(r["O"] for r in reqs)
    out = dict(makespan=t, tok_s=toks / (t - min(r["arrive"] for r in reqs)), ttft_mean=ttft.mean(), ttft_p50=np.median(ttft),
               ttft_p99=np.percentile(ttft, 99), lat_mean=lat.mean(), lat_p50=np.median(lat), lat_p99=np.percentile(lat, 99),
               gap_mean=gaps.mean(), gap_max=gaps.max(), gap_p99=np.percentile(gaps, 99), preempt=preempt,
               pad_waste=(1 - used_slots / pad_slots) if pad_slots else 0.0)
    s = [r for r in done if r.get("sentence")]
    if s: out.update(sentence_ttft=s[0]["first"] - s[0]["arrive"], sentence_lat=s[0]["end"] - s[0]["arrive"])
    if peak_reserved:
        res = np.array(peak_reserved); out.update(mean_reserved=res[:, 0].mean() * page, mean_used=res[:, 1].mean())
    return out
