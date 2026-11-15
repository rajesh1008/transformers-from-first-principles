"""A configuration reader: turn a model's configuration into the eight-or-so numbers that decide its
size, then count its learned numbers part by part, the way Chapter 16 does on paper.
Three families are understood: GPT-2 (learned positions, layer norm with bias, biases everywhere,
plain feed-forward), Llama-style (rotary, RMSNorm, no biases, gated feed-forward, grouped key-value
heads), and Qwen3 (Llama-style plus a separate head width and a norm on each query and key).
Tiny's own settings use the same reader through tiny_spec()."""

def spec(cfg):
    """Read a Hugging Face config.json (a dict) into one plain dict of named settings."""
    t = cfg.get("model_type")
    if t == "gpt2":
        d = cfg["n_embd"]
        return dict(family="gpt2", vocab=cfg["vocab_size"], d=d, layers=cfg["n_layer"],
                    heads=cfg["n_head"], kv_heads=cfg["n_head"], head_dim=d // cfg["n_head"],
                    d_ff=cfg.get("n_inner") or 4 * d, gated=False, positions=cfg["n_positions"],
                    learned_positions=True, norm="layer", bias=True, qk_norm=False,
                    tied=cfg.get("tie_word_embeddings", True), context=cfg["n_positions"])
    if t in ("llama", "qwen3"):
        d, h = cfg["hidden_size"], cfg["num_attention_heads"]
        return dict(family=t, vocab=cfg["vocab_size"], d=d, layers=cfg["num_hidden_layers"],
                    heads=h, kv_heads=cfg.get("num_key_value_heads", h),
                    head_dim=cfg.get("head_dim") or d // h, d_ff=cfg["intermediate_size"],
                    gated=True, positions=0, learned_positions=False, norm="rms",
                    bias=cfg.get("attention_bias", False), qk_norm=(t == "qwen3"),
                    tied=cfg.get("tie_word_embeddings", False), context=cfg["max_position_embeddings"])
    raise ValueError("unknown model_type " + str(t))

def tiny_spec(vocab_size=2048, n_layers=12, d_model=128, n_heads=4, context=1024, d_ff=None,
              norm="layer", tied=True):
    """Tiny's constructor arguments as the same dict (Tiny's attention has no biases; its
    feed-forward block and its layer norms do; RMSNorm has a gain only)."""
    return dict(family="tiny", vocab=vocab_size, d=d_model, layers=n_layers, heads=n_heads,
                kv_heads=n_heads, head_dim=d_model // n_heads, d_ff=d_ff or 4 * d_model,
                gated=False, positions=context, learned_positions=True, norm=norm,
                bias="ffn-only", qk_norm=False, tied=tied, context=context)

def count(s):
    """Learned numbers by part, from the settings alone.  Returns an ordered list of
    (part, formula as text, number)."""
    d, L, V = s["d"], s["layers"], s["vocab"]
    hq, hk, hd, f = s["heads"], s["kv_heads"], s["head_dim"], s["d_ff"]
    norm_size = 2 * d if s["norm"] == "layer" else d          # gain and bias, or gain only
    rows = [("token table", f"{V:,} x {d}", V * d)]
    if s["learned_positions"]:
        rows.append(("position table", f"{s['positions']:,} x {d}", s["positions"] * d))
    q, kv, o = d * hq * hd, d * hk * hd, hq * hd * d
    att = q + 2 * kv + o
    att_txt = f"{d}x{hq*hd} + 2 x {d}x{hk*hd} + {hq*hd}x{d}"
    if s["bias"] is True:
        att += hq * hd + 2 * hk * hd + d
        att_txt += f" + biases {hq*hd + 2*hk*hd + d}"
    if s["qk_norm"]:
        att += 2 * hd
        att_txt += f" + q/k norms 2x{hd}"
    ffn = (3 if s["gated"] else 2) * d * f
    ffn_txt = f"{3 if s['gated'] else 2} x {d}x{f}"
    if s["bias"] in (True, "ffn-only"):
        ffn += f + d
        ffn_txt += f" + biases {f + d}"
    norms = 2 * norm_size
    per_layer = att + ffn + norms
    rows += [("attention, per layer", att_txt, att), ("feed-forward, per layer", ffn_txt, ffn),
             ("norms, per layer", f"2 x {norm_size}", norms),
             (f"all {L} layers", f"{L} x {per_layer:,}", L * per_layer),
             ("final norm", f"{norm_size}", norm_size)]
    if not s["tied"]:
        rows.append(("output table (untied)", f"{d} x {V:,}", d * V))
    return rows

def total(s):
    rows = count(s)
    return sum(n for part, _, n in rows if "per layer" not in part)

def non_embedding(s):
    """The learned numbers in the layers and the final norm (positions and tokens left out)."""
    d = s["d"]
    rows = dict((p, n) for p, _, n in count(s))
    return rows[f"all {s['layers']} layers"] + rows["final norm"]

def kv_numbers_per_token(s):
    """Numbers kept per token in the key-value cache: a key and a value per layer per key-value head."""
    return 2 * s["layers"] * s["kv_heads"] * s["head_dim"]
