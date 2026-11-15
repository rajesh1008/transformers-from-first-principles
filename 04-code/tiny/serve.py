"""tiny/serve.py (Chapter 31): serving pieces for any GPT2 from tiny/load.py.

  decode_step(model, caches, tokens, positions)
        one decode step for a batch of B conversations at once.  The linear maps (query/key/value, output map,
        feed-forward block, output scores) run on a [B, d] table, so each weight table is read once for the whole
        batch; attention runs per conversation over its own cache, so contexts may differ (ragged batching).
        caches[i] is a tiny/cache.py cache (list of (K, V) per layer); each is extended in place.  Returns [B, vocab].
  prefill(model, ids)  -> (scores of last position, cache)
  PagedKV(num_pages, page_tokens)
        a page allocator for cache slots: alloc(seq, n_tokens) grows a sequence's block table one page at a time,
        free(seq) returns its pages; used_slots / reserved_slots measure what is written against what is held.
  contiguous_reserved(lengths_max)  slots a contiguous allocator holds: every sequence reserves its maximum."""
import math


def prefill(model, ids):
    import torch
    from tiny.cache import new_cache
    cache = new_cache(model)
    with torch.no_grad():
        s = model.forward(list(ids), cache, 0)[-1]
    return s, cache


def decode_step(model, caches, tokens, positions):
    import torch
    W, H, d = model.W, model.H, model.d
    B = len(tokens); hd = d // H
    with torch.no_grad():
        x = W["wte.weight"][torch.as_tensor(tokens)] + W["wpe.weight"][torch.as_tensor(positions)]
        for l in range(model.L):
            p = f"h.{l}."
            q, k, v = (model._ln(x, p + "ln_1") @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]).split(d, -1)
            q, k, v = (z.view(B, H, 1, hd) for z in (q, k, v))
            outs = []
            for i in range(B):
                K, V = caches[i][l]
                K = torch.cat([K, k[i]], 1); V = torch.cat([V, v[i]], 1)
                caches[i][l] = (K, V)
                s = q[i] @ K.transpose(-1, -2) / math.sqrt(hd)          # [H, 1, S]
                outs.append((torch.softmax(s, -1) @ V).reshape(1, d))
            o = torch.cat(outs, 0)
            x = x + o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
            f = model._gelu(model._ln(x, p + "ln_2") @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
            x = x + f @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
        return model._ln(x, "ln_f") @ W["wte.weight"].T


class PagedKV:
    def __init__(self, num_pages, page_tokens):
        self.page_tokens = page_tokens
        self.free_pages = list(range(num_pages - 1, -1, -1))
        self.tables, self.lengths = {}, {}

    def can_alloc(self, seq, n=1):
        have = len(self.tables.get(seq, [])) * self.page_tokens
        need = math.ceil(max(0, self.lengths.get(seq, 0) + n - have) / self.page_tokens)
        return need <= len(self.free_pages)

    def alloc(self, seq, n=1):
        """Record n more tokens for seq; take new pages only when the last one is full.  False if out of pages."""
        if not self.can_alloc(seq, n):
            return False
        t = self.tables.setdefault(seq, []); self.lengths[seq] = self.lengths.get(seq, 0) + n
        while len(t) * self.page_tokens < self.lengths[seq]:
            t.append(self.free_pages.pop())
        return True

    def free(self, seq):
        self.free_pages.extend(reversed(self.tables.pop(seq, []))); self.lengths.pop(seq, None)

    def used_slots(self):
        return sum(self.lengths.values())

    def reserved_slots(self):
        return sum(len(t) for t in self.tables.values()) * self.page_tokens


def contiguous_reserved(lengths_max):
    return sum(lengths_max)
