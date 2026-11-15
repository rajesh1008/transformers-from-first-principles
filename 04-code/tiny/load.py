"""tiny/load.py (Chapter 29): loading released weights into the book's own forward pass, and checking them.

  read_header(path)              the safetensors header: {name: {dtype, shape, data_offsets}}, metadata, header bytes
  learned_count(header)          numbers that are learned (GPT-2's causal-mask buffers, 'attn.bias' of rank 4, excluded)
  GPT2.from_dir(d)               GPT-2-shaped files (openai-community/gpt2 names; Conv1D maps stored [in, out])
  Llama.from_dir(d)              Llama-shaped files (RMSNorm, rotary half-split pairs, grouped key-value heads, SwiGLU;
                                 nn.Linear maps stored [out, in]); SmolLM2 is one
      .scores(ids)               raw next-token scores at every position, float32 torch [T, vocab]
      .forward(ids, cache, start)  the same, with an optional key-value cache (Chapter 30; see tiny/cache.py)
  compare(S, R)                  teacher-forced comparison of two score tables: largest |difference| in log-probabilities,
                                 mean |difference| in scores, same top token at every position
  tolerance(dtype)               the comparison bar set from the number format before looking (32-bit: 1e-4)
  render_chatml(messages, system, add_generation_prompt)
                                 the ChatML text SmolLM2-Instruct's template produces, written out by hand
Everything is 32-bit on the CPU unless the caller casts the tables.  Both classes expose `scores`, so tiny/generate.py
runs on them unchanged."""
import json, math, os, struct
import numpy as np


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    meta = head.pop("__metadata__", None)
    return head, meta, n


def learned_count(header):
    return sum(int(np.prod(v["shape"])) for k, v in header.items()
               if not (k.endswith("attn.bias") and len(v["shape"]) == 4))


def _tables(path):
    import torch
    from safetensors.torch import load_file
    return {k: v.float() for k, v in load_file(path).items()}


class GPT2:
    def __init__(self, W, cfg):
        self.W, self.cfg = W, cfg
        self.L, self.H, self.d = cfg["n_layer"], cfg["n_head"], cfg["n_embd"]
        self.eps = cfg["layer_norm_epsilon"]

    @classmethod
    def from_dir(cls, d):
        return cls(_tables(os.path.join(d, "model.safetensors")), json.load(open(os.path.join(d, "config.json"))))

    def _ln(self, x, p):
        import torch
        m = x.mean(-1, keepdim=True); v = ((x - m) ** 2).mean(-1, keepdim=True)
        return (x - m) / torch.sqrt(v + self.eps) * self.W[p + ".weight"] + self.W[p + ".bias"]

    @staticmethod
    def _gelu(x):
        import torch
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

    def forward(self, ids, cache=None, start=0):
        """Scores for the new tokens `ids`, which sit at positions start, start + 1, ...  If `cache` is a list of
        (K, V) pairs (one per layer, each [H, T_past, d/H] or None), the new keys and values are appended to it and the
        new queries attend over past and new.  With cache=None this is the plain full pass of Chapter 29."""
        import torch
        W, H, d = self.W, self.H, self.d
        T = len(ids); hd = d // H
        x = W["wte.weight"][torch.as_tensor(ids)] + W["wpe.weight"][start:start + T]
        for l in range(self.L):
            p = f"h.{l}."
            q, k, v = (self._ln(x, p + "ln_1") @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]).split(d, -1)
            q, k, v = (z.view(T, H, hd).transpose(0, 1) for z in (q, k, v))
            if cache is not None:
                if cache[l] is not None:
                    k = torch.cat([cache[l][0], k], 1); v = torch.cat([cache[l][1], v], 1)
                cache[l] = (k, v)
            S = k.shape[1]                                   # keys visible: past + new
            mask = torch.ones(T, S, dtype=torch.bool).tril(S - T)
            s = (q @ k.transpose(-1, -2) / math.sqrt(hd)).masked_fill(~mask, float("-inf"))
            o = (torch.softmax(s, -1) @ v).transpose(0, 1).reshape(T, d)
            x = x + o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
            f = self._gelu(self._ln(x, p + "ln_2") @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
            x = x + f @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
        return self._ln(x, "ln_f") @ W["wte.weight"].T

    def scores(self, ids):
        import torch
        with torch.no_grad():
            return self.forward(list(ids))


class Llama:
    def __init__(self, W, cfg):
        self.W, self.cfg = W, cfg
        self.L, self.d = cfg["num_hidden_layers"], cfg["hidden_size"]
        self.Hq, self.Hkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.hd = self.d // self.Hq
        self.eps, self.theta = cfg["rms_norm_eps"], cfg["rope_theta"]
        self.emb = W["model.embed_tokens.weight"]
        self.out = W.get("lm_head.weight", self.emb)       # tied when the file has no output table

    @classmethod
    def from_dir(cls, d):
        return cls(_tables(os.path.join(d, "model.safetensors")), json.load(open(os.path.join(d, "config.json"))))

    def _rms(self, x, w):
        import torch
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) * w

    def _rope(self, x, pos):
        import torch
        hd = self.hd
        inv = 1.0 / self.theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)
        ang = pos[:, None].float() * inv[None]
        c, s = torch.cos(ang), torch.sin(ang)
        x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)

    def forward(self, ids, cache=None, start=0):
        import torch
        W, T, hd, Hq, Hkv = self.W, len(ids), self.hd, self.Hq, self.Hkv
        pos = torch.arange(start, start + T)
        x = self.emb[torch.as_tensor(ids)]
        for l in range(self.L):
            p = f"model.layers.{l}."
            a = self._rms(x, W[p + "input_layernorm.weight"])
            q = (a @ W[p + "self_attn.q_proj.weight"].T).view(T, Hq, hd).transpose(0, 1)
            k = (a @ W[p + "self_attn.k_proj.weight"].T).view(T, Hkv, hd).transpose(0, 1)
            v = (a @ W[p + "self_attn.v_proj.weight"].T).view(T, Hkv, hd).transpose(0, 1)
            q, k = self._rope(q, pos), self._rope(k, pos)
            if cache is not None:
                if cache[l] is not None:
                    k = torch.cat([cache[l][0], k], 1); v = torch.cat([cache[l][1], v], 1)
                cache[l] = (k, v)                            # the cache holds Hkv heads, not Hq
            S = k.shape[1]
            kk, vv = k.repeat_interleave(Hq // Hkv, 0), v.repeat_interleave(Hq // Hkv, 0)
            mask = torch.ones(T, S, dtype=torch.bool).tril(S - T)
            s = (q @ kk.transpose(-1, -2) / math.sqrt(hd)).masked_fill(~mask, float("-inf"))
            o = (torch.softmax(s, -1) @ vv).transpose(0, 1).reshape(T, self.d)
            x = x + o @ W[p + "self_attn.o_proj.weight"].T
            m = self._rms(x, W[p + "post_attention_layernorm.weight"])
            x = x + (torch.nn.functional.silu(m @ W[p + "mlp.gate_proj.weight"].T) * (m @ W[p + "mlp.up_proj.weight"].T)) \
                @ W[p + "mlp.down_proj.weight"].T
        return self._rms(x, W["model.norm.weight"]) @ self.out.T

    def scores(self, ids):
        import torch
        with torch.no_grad():
            return self.forward(list(ids))


def compare(S, R):
    import torch
    a, b = torch.log_softmax(S.double(), -1), torch.log_softmax(R.double(), -1)
    return dict(max_logp=float((a - b).abs().max()), mean_score=float((S - R).abs().mean()),
                same_top=bool((S.argmax(-1) == R.argmax(-1)).all()))


def tolerance(dtype="float32"):
    """Largest |difference| in log-probabilities accepted between two correct implementations, set before looking:
    32-bit about 7 significant digits on scores of size ~100 gives ~1e-5; the bar is ten times that."""
    return {"float32": 1e-4, "bfloat16": 1.0, "float16": 0.5}[str(dtype).replace("torch.", "")]


def render_chatml(messages, system="You are a helpful AI assistant named SmolLM, trained by Hugging Face",
                  add_generation_prompt=True):
    out = ""
    if messages and messages[0]["role"] != "system" and system is not None:
        out += f"<|im_start|>system\n{system}<|im_end|>\n"
    for m in messages:
        out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    if add_generation_prompt:
        out += "<|im_start|>assistant\n"
    return out
