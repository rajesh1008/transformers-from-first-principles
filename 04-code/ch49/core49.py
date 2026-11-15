"""Chapter 49: the capstone as a served artefact.  Loads a model from a safetensors file written by export49.py (32-bit, or 8-bit
with one scale per output row), and runs Chapter 48's cached generation loop (eval48.py's step_cached) one request at a time.
CPU, one thread, 32-bit arithmetic."""
import os, sys, json, hashlib, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from tiny.capstone.model import Capstone
from safetensors.torch import load_file
torch.set_num_threads(1)
ART = os.path.join(HERE, "artefact")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def load_model(kind="fp32"):
    cfg = json.load(open(os.path.join(ART, "config.json")))
    m = Capstone(**cfg["model"]); t = load_file(os.path.join(ART, f"capstone-{kind}.safetensors")); state = {}
    for n, p in m.state_dict().items():
        state[n] = t[n + ".q"].float() * t[n + ".scale"] if n + ".q" in t else t[n]
    m.load_state_dict(state); m.eval(); return m, cfg

def step_cached(m, ids, cache):
    """One position (or a whole prompt) through the model, keeping each layer's keys and values (as eval48.py)."""
    pos0 = cache[0][0].shape[2] if cache else 0; n = ids.shape[1]
    x = m.E(ids) + m.P(torch.arange(pos0, pos0 + n)); new = []
    for li, b in enumerate(m.blocks):
        d = x.shape[-1]; q, k, v = b.qkv(b.ln1(x)).split(d, -1)
        q, k, v = (t.view(1, n, b.h, d // b.h).transpose(1, 2) for t in (q, k, v))
        if cache: k = torch.cat([cache[li][0], k], 2); v = torch.cat([cache[li][1], v], 2)
        new.append((k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=(pos0 == 0)).transpose(1, 2).reshape(1, n, d)
        x = x + b.o(a); x = x + b.down(F.gelu(b.up(b.ln2(x)), approximate="tanh"))
    return m.lnf(x) @ m.E.weight.T, new

def generate(m, prompt_ids, max_tokens, temperature, seed, ctx=256):
    """Yields (token id, seconds since start) for each new token.  Stops at max_tokens or a full window."""
    g = torch.Generator().manual_seed(seed); t0 = time.perf_counter(); ids = list(prompt_ids)
    with torch.no_grad():
        lg, cache = step_cached(m, torch.tensor([ids]), None)
        for _ in range(max_tokens):
            p = torch.softmax(lg[0, -1] / max(temperature, 1e-6), -1)
            nxt = int(torch.multinomial(p, 1, generator=g)) if temperature > 0 else int(p.argmax())
            ids.append(nxt); yield nxt, time.perf_counter() - t0
            if len(ids) >= ctx: return
            lg, cache = step_cached(m, torch.tensor([[nxt]]), cache)

def first_token_logprobs(m, ids, k=None):
    with torch.no_grad():
        lp = torch.log_softmax(m(torch.tensor([ids]))[0, -1], -1)
    return lp if k is None else lp[k]
