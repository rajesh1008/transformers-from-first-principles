"""Lab 16: a configuration reader, three ways to misread a configuration that raise no error, and a
checkpoint.  Every breakage produces a plausible number; only a check against something independent
(the weights file's header, Tiny's own count, or the config's other fields) catches it."""
from _common import load_config, load_header, save_json, c
from reader import spec, tiny_spec, total, kv_numbers_per_token
from tiny.model import Tiny
import math

def file_learned(name, s):
    h = load_header(name)["tables"]
    return sum(math.prod(v["shape"]) for k, v in h.items()
               if k.split(".")[-2:] != ["attn", "bias"] and not (s["tied"] and k == "lm_head.weight"))
def file_all(name):
    return sum(math.prod(v["shape"]) for v in load_header(name)["tables"].values())

print("## Part 1: the reader on three files")
res = {}
for name in ["GPT-2 small", "SmolLM2-135M", "Qwen3-0.6B"]:
    s = spec(load_config(name))
    res[name] = total(s)
    print(f"  {name:13s} hand {c(total(s)):>12s}  file (learned tables) {c(file_learned(name, s)):>12s}"
          f"  key-value numbers per token {c(kv_numbers_per_token(s))}")

print("\n## Part 2: three misreadings")
q = spec(load_config("Qwen3-0.6B"))
b1 = dict(q, tied=False)
print(f"  Breakage 1, tie_word_embeddings ignored (Qwen3-0.6B): {c(total(b1))}"
      f"  vs every table in the file {c(file_all('Qwen3-0.6B'))}  -> agrees with the file: {total(b1) == file_all('Qwen3-0.6B')}")
print(f"    the right count is {c(total(q))}; the misreading adds {c(total(b1) - total(q))} ({(total(b1) / total(q) - 1) * 100:.1f} per cent)")
b2 = dict(q, head_dim=q["d"] // q["heads"])
print(f"  Breakage 2, head width taken as hidden_size / num_attention_heads = {q['d']} / {q['heads']} = {b2['head_dim']}"
      f" (the config says head_dim {q['head_dim']}):")
print(f"    count {c(total(b2))} ({c(total(q) - total(b2))} short); key-value numbers per token {c(kv_numbers_per_token(b2))}"
      f" instead of {c(kv_numbers_per_token(q))}")
sm = spec(load_config("SmolLM2-135M"))
b3 = dict(sm, kv_heads=sm["heads"])
print(f"  Breakage 3, num_key_value_heads overlooked (SmolLM2-135M): count {c(total(b3))} ({c(total(b3) - total(sm))} too many);")
print(f"    key-value cache at {c(sm['context'])} tokens in 16-bit {kv_numbers_per_token(b3) * 2 * sm['context'] / 1e6:.1f} MB"
      f" instead of {kv_numbers_per_token(sm) * 2 * sm['context'] / 1e6:.1f} MB")

print("\n## Part 3: the checkpoint")
tests = [
    ("the reader counts default Tiny as Tiny.count() does (2,766,592)", total(tiny_spec()) == Tiny().n_params == 2766592),
    ("the reader counts GPT-2 small as 124,439,808 and the file's learned tables agree",
     res["GPT-2 small"] == 124439808 == file_learned("GPT-2 small", spec(load_config("GPT-2 small")))),
    ("SmolLM2-135M: hand count equals every number in the file (134,515,008)", res["SmolLM2-135M"] == file_all("SmolLM2-135M") == 134515008),
    ("Qwen3-0.6B: hand count 596,049,920; the file holds 155,582,464 more (a stored copy of the tied table)",
     res["Qwen3-0.6B"] == 596049920 and file_all("Qwen3-0.6B") - res["Qwen3-0.6B"] == 155582464),
    ("breakage 1 agrees with the file and is still wrong", total(b1) == file_all("Qwen3-0.6B") != total(q)),
    ("breakage 2 halves Qwen3-0.6B's key-value cache estimate", kv_numbers_per_token(b2) * 2 == kv_numbers_per_token(q)),
    ("breakage 3 triples SmolLM2-135M's key-value cache estimate", kv_numbers_per_token(b3) == 3 * kv_numbers_per_token(sm)),
]
for i, (t, ok) in enumerate(tests, 1):
    print(f"  {i}. {t}: {'pass' if ok else 'FAIL'}")
print("all checkpoint tests pass:", all(ok for _, ok in tests))
save_json("lab16", {t: bool(ok) for t, ok in tests})
