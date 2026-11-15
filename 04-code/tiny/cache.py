"""tiny/cache.py (Chapter 30): the key-value cache, for any model in tiny/load.py (GPT2, Llama) whose forward accepts
(ids, cache, start).

  new_cache(model)                 one empty slot per layer
  cache_numbers(cache), cache_bytes(cache)   what the cache holds now (numbers; bytes at the tables' format)
  kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_number)
                                   the hand formula: 2 (keys and values) x layers x kv_heads x head_dim x bytes
  greedy(model, ids, n, cached=True)
                                   greedy continuation of n tokens; with cached=True one prefill pass over the prompt,
                                   then one single-token pass per new token at the right position; with cached=False a
                                   full pass over everything for every token (Chapters 27 to 29).  Returns (ids, per-step
                                   raw score rows)."""
import numpy as np


def new_cache(model):
    return [None] * model.L


def cache_numbers(cache):
    return sum(int(k.numel() + v.numel()) for k, v in (c for c in cache if c is not None))


def cache_bytes(cache):
    return sum(int(k.numel() * k.element_size() + v.numel() * v.element_size()) for k, v in (c for c in cache if c is not None))


def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_number):
    return 2 * layers * kv_heads * head_dim * bytes_per_number


def greedy(model, ids, n, cached=True):
    import torch
    ids = list(ids); rows = []
    with torch.no_grad():
        if cached:
            cache = new_cache(model)
            s = model.forward(ids, cache, 0)[-1]
            for _ in range(n):
                rows.append(s); nxt = int(s.argmax()); ids.append(nxt)
                s = model.forward([nxt], cache, len(ids) - 1)[-1]
        else:
            for _ in range(n):
                s = model.forward(ids)[-1]
                rows.append(s); ids.append(int(s.argmax()))
    return ids, rows
