"""Count each model's learned numbers by hand from its configuration, then check the count against
the tables actually stored in its weights file (the header fetched by fetch.py) and, for Tiny,
against Tiny.count().  Prints every step."""
import math
from _common import load_config, load_header, save_json, MODELS, c
from reader import spec, tiny_spec, count, total, non_embedding
from tiny.model import Tiny

out = {}
def show(name, s):
    print(f"\n{name}: vocab {s['vocab']:,}, layers {s['layers']}, width {s['d']}, heads {s['heads']} "
          f"(key-value heads {s['kv_heads']}, head width {s['head_dim']}), d_ff {s['d_ff']:,}, "
          f"{'gated' if s['gated'] else 'plain'} feed-forward, {s['norm']} norm, "
          f"{'learned' if s['learned_positions'] else 'rotary'} positions, "
          f"{'tied' if s['tied'] else 'untied'} output")
    for part, formula, n in count(s):
        print(f"  {part:26s} {formula:42s} {c(n):>14s}")
    t = total(s)
    print(f"  {'TOTAL (hand)':26s} {'':42s} {c(t):>14s}   non-embedding {c(non_embedding(s))}")
    return t

# Tiny and the hand Tiny first: the settings are constructor arguments, the check is Tiny.count()
for name, kw in [("Tiny (default)", {}),
                 ("hand Tiny", dict(vocab_size=12, n_layers=2, d_model=2, n_heads=1, d_ff=3,
                                    context=6, norm="rms"))]:
    s = tiny_spec(**{k: v for k, v in kw.items()})
    t = show(name, s)
    m = Tiny(**kw, **({"activation": "relu"} if kw else {}))
    print(f"  Tiny.count() from the built tables: {c(m.n_params)}  match: {t == m.n_params}")
    out[name] = dict(hand=t, code=m.n_params)

for name in MODELS:
    s = spec(load_config(name))
    t = show(name, s)
    h = load_header(name)
    stored = {k: math.prod(v["shape"]) for k, v in h["tables"].items()}
    every = sum(stored.values())
    # GPT-2's file also stores each layer's causal mask as h.N.attn.bias, a 1 x 1 x 1024 x 1024 table of
    # ones and zeros that is never learned.  Match the last two name parts exactly: a filter on the
    # ending "attn.bias" also catches h.N.attn.c_attn.bias, which IS learned (the first run of this
    # script made that mistake and came out 27,648 short, as Chapter 14's did).
    def is_mask(k):
        return k.split(".")[-2:] == ["attn", "bias"]
    learned = {k: n for k, n in stored.items() if not is_mask(k)
               and not (s["tied"] and k == "lm_head.weight")}
    dtypes = sorted({v["dtype"] for v in h["tables"].values()})
    print(f"  weights file: {len(stored)} tables, {c(every)} numbers stored ({','.join(dtypes)}), "
          f"{c(h['data_bytes'])} bytes")
    extra = {k: n for k, n in stored.items() if k not in learned}
    if extra:
        kinds = sorted({"h.N.attn.bias (mask)" if is_mask(k) else k for k in extra})
        print(f"  stored but not learned: {len(extra)} tables, {c(sum(extra.values()))} numbers ({', '.join(kinds)})")
    print(f"  learned tables in the file: {c(sum(learned.values()))}   hand count matches: {sum(learned.values()) == t}")
    out[name] = dict(hand=t, file_numbers=every, file_learned=sum(learned.values()),
                     file_bytes=h["data_bytes"], non_embedding=non_embedding(s), dtypes=dtypes)
save_json("count", out)
