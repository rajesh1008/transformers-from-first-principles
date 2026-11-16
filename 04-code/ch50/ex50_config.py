"""Exercise 50.12, configuration-only comparison with SmolLM3-3B. No weights downloaded or generation claimed.
Source: https://huggingface.co/HuggingFaceTB/SmolLM3-3B/resolve/main/config.json, fetched 16 November 2026.
Parameter count is made on the meta device, not by allocating or loading trained weights."""
import json, os, torch
from transformers import AutoConfig, AutoModelForCausalLM
HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, "ckpt", "smollm3-config"); os.makedirs(p, exist_ok=True)
import urllib.request
for name, url in [("config.json", "https://huggingface.co/HuggingFaceTB/SmolLM3-3B/resolve/main/config.json"),
                  ("model-info.json", "https://huggingface.co/api/models/HuggingFaceTB/SmolLM3-3B")]:
    if not os.path.exists(os.path.join(p, name)): urllib.request.urlretrieve(url, os.path.join(p, name))
c = AutoConfig.from_pretrained(p)
with torch.device("meta"): m = AutoModelForCausalLM.from_config(c)
r = dict(model="HuggingFaceTB/SmolLM3-3B", parameters=sum(p.numel() for p in m.parameters()), config=c.to_dict(), weights_loaded=False)
info = json.load(open(os.path.join(p, "model-info.json"))); r["hub_createdAt"] = info.get("createdAt")
json.dump(r, open(os.path.join(HERE, "outputs", "ex50_config.json"), "w"), indent=1)
print(r['model'], r['parameters'], 'parameters, Hub creation', r['hub_createdAt'])
print(f'{c.num_hidden_layers} x {c.hidden_size}, {c.num_attention_heads}/{c.num_key_value_heads} heads, vocab {c.vocab_size}, context {c.max_position_embeddings}')
