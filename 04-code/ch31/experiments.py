"""Chapter 31: simulator experiments on the measured GPT-2 step costs (no model calls; seconds to run).
Part A: throughput against latency as the arrival rate grows (static, continuous, chunked), 200 requests.
Part B: queue order, FCFS against shortest-reply-first, at a loaded rate.  Part C: chunked prefill and the longest gap
between tokens.  Part D: cache memory, contiguous reservation against pages.  Part E: cost per million tokens."""
from _common import *
import numpy as np
from workload import requests, cost_model
from sim import simulate
from tiny.serve import PagedKV, contiguous_reserved
dec, pre = cost_model(); out = {}
print("Part A: 200 requests, B up to 16; output tokens/s and reply latency (s) against arrival rate (requests/s)")
rows = []
for rate in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
    R = requests(200, rate)
    for sch in ("static", "continuous", "chunked"):
        m = simulate(R, dec, pre, sch, B=16, chunk=128)
        rows.append(dict(rate=rate, scheme=sch, **{k: float(v) for k, v in m.items()}))
        print(f"  rate {rate:4.2f} {sch:10s}: {m['tok_s']:5.1f} tokens/s; first token p50 {m['ttft_p50']:6.2f} p99 {m['ttft_p99']:7.2f}; "
              f"reply p50 {m['lat_p50']:6.2f} p99 {m['lat_p99']:7.2f}; longest gap {m['gap_max']:5.2f}")
out["A"] = rows
print("\nPart B: queue order at rate 1.5, continuous, B 16 (oracle reply lengths for shortest-first)")
R = requests(200, 1.5)
for order in ("fcfs", "short"):
    m = simulate(R, dec, pre, "continuous", B=16, order=order)
    lat = m
    print(f"  {order:5s}: reply mean {m['lat_mean']:6.2f} p50 {m['lat_p50']:6.2f} p99 {m['lat_p99']:7.2f}; first token p99 {m['ttft_p99']:6.2f}; {m['tok_s']:5.1f} tokens/s")
    out["B_" + order] = {k: float(v) for k, v in m.items()}
print("\nPart C: chunk size against the gap between tokens, rate 1.0, B 16")
for ch in (32, 64, 128, 256, 512):
    m = simulate(requests(200, 1.0), dec, pre, "chunked", B=16, chunk=ch)
    print(f"  chunk {ch:4d}: gap mean {m['gap_mean'] * 1000:6.0f} ms, p99 {m['gap_p99'] * 1000:6.0f} ms, max {m['gap_max'] * 1000:6.0f} ms; first token p50 {m['ttft_p50']:5.2f} s; {m['tok_s']:5.1f} tokens/s")
    out[f"C_{ch}"] = {k: float(v) for k, v in m.items()}
m = simulate(requests(200, 1.0), dec, pre, "continuous", B=16)
print(f"  unchunked: gap mean {m['gap_mean'] * 1000:6.0f} ms, p99 {m['gap_p99'] * 1000:6.0f} ms, max {m['gap_max'] * 1000:6.0f} ms; first token p50 {m['ttft_p50']:5.2f} s; {m['tok_s']:5.1f} tokens/s")
print("\nPart D: cache memory.  The twenty requests, each allowed up to 128 new tokens")
R = requests()
cont = contiguous_reserved([r["P"] + 128 for r in R]); used = sum(r["P"] + r["O"] for r in R)
print(f"  if all twenty were resident at their final lengths: written {used} slots; contiguous reservation {cont} ({1 - used / cont:.1%} never written)")
for pg in (1, 4, 16, 64, 256):
    kv = PagedKV(10 ** 6, pg)
    for r in R: kv.alloc(r["id"], r["P"] + r["O"])
    print(f"  pages of {pg:3d}: reserved {kv.reserved_slots():5d} ({1 - used / kv.reserved_slots():.1%} of reserved slots empty, all in last pages)")
print("  admission at rate 3.0, 200 requests, cache of 256 pages x 16 slots = 4,096 slots (both schemes, B up to 64)")
R = requests(200, 3.0)
for alloc in ("contig", "paged"):
    m = simulate(R, dec, pre, "continuous", B=64, mem=256, page=16, alloc=alloc)
    print(f"  {alloc:6s}: {m['tok_s']:5.1f} tokens/s; reply p50 {m['lat_p50']:6.2f} p99 {m['lat_p99']:6.2f}; mean slots reserved {m['mean_reserved']:6.0f} written {m['mean_used']:6.0f}; pre-emptions {m['preempt']}")
    out["D_" + alloc] = {k: float(v) for k, v in m.items()}
print("\nPart E: cost per million new tokens on this CPU thread, from step_cost.py's measured throughput")
D = load_json("step_cost")
for r in D["B"]:
    tps = r["B"] / r["step"]
    print(f"  context {r['ctx']:4d}, B {r['B']:3d}: {tps:6.1f} tokens/s -> {1e6 / tps / 3600:6.2f} machine-hours per million tokens "
          f"= ${1e6 / tps / 3600:6.2f} per million at $1 per machine-hour")
save_json("experiments", out)
