"""Chapter 31: the twenty-request workload (seed 1865) and the step-cost model fitted in step_cost.py."""
import numpy as np
from _common import load_json, SEED

def cost_model():
    D = load_json("step_cost")["D"]; a, b, c = D["dec"]; pa, pb, pc = D["pre"]
    dec = lambda B, ctx_sum: a + b * B + c * ctx_sum if B else 0.0
    pre = lambda P, past=0: pa + pb * P + pc * P * (P + 2 * past)     # past: tokens already cached (chunked prefill)
    return dec, pre

def requests(n=20, rate=2.0, seed=SEED, sentence_len=6):
    """n requests: arrival gaps exponential at `rate` per second; prompt lengths 16 to 512 (log-uniform); reply lengths
    8 to 128 (log-uniform).  Request 0 is the running sentence: 6 prompt tokens, 16 new tokens, arriving third."""
    r = np.random.default_rng(seed)
    t = np.cumsum(r.exponential(1 / rate, n)); P = np.exp(r.uniform(np.log(16), np.log(512), n)).astype(int)
    O = np.exp(r.uniform(np.log(8), np.log(128), n)).astype(int)
    reqs = [dict(id=i, arrive=float(t[i]), P=int(P[i]), O=int(O[i])) for i in range(n)]
    reqs[2].update(P=sentence_len, O=16, sentence=True)
    return reqs
