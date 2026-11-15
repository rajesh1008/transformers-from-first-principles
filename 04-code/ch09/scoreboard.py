"""Chapter 9 scoreboard and Tiny's multi-head attention on the running sentence.
Tiny now has MultiHead(n_heads=4): four heads of 32 and an output table.
Run from 04-code/ch09:  python3 scoreboard.py   (--measure to time your own machine)"""
from _common import *
import time
from tiny.attention import MultiHead, entropy
from tiny.embed import Embedding, PositionTable
from tiny.tokenizer import load_bpe
from tiny.corpus import load_text

tok = load_bpe(TINY_BPE)
emb, pos, attn = Embedding(len(tok)), PositionTable(), MultiHead(n_heads=4)
ids = tok.encode(SENTENCE)
B = emb(ids) + pos(range(len(ids)))
out, A = attn(B, return_shares=True)
print("Tiny's running sentence:", [tok.piece(i) for i in ids], "->", B.shape, "->", out.shape, "| shares", A.shape)
print("untrained: every head's last row is an even split:", bool(np.allclose(A[:, -1], 1 / 7, atol=1e-4)))
novel = np.array(tok.encode(load_text()))
windows = [novel[i:i + 1024] for i in range(0, len(novel), 1024)]
def measure():
    t = time.perf_counter()
    for w in windows:
        attn(emb(w) + pos(range(len(w))))
    ms = (time.perf_counter() - t) * 1000
    return {"ms": ms, "tokens": int(len(novel))}
r = recorded("scoreboard", measure)
print(f"four heads over the whole novel ({r['tokens']:,} tokens, {len(windows)} windows): {r['ms']:.0f} ms"
      f" ({r['tokens'] / r['ms'] * 1000:,.0f} tokens per second)  (measured {r['measured']})")
learned = emb.table.size + pos.table.size + attn.n_params
print(f"learned numbers: 262,144 (tokens, tied) + 131,072 (positions) + 49,152 (four heads: query, key, value)"
      f" + 16,384 (output table) = {learned:,}")
print("shares per full window: 4 x 1,024 x 1,024 =", f"{4 * 1024 * 1024:,}")
