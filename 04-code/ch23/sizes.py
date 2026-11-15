"""The five Tinys of Chapter 23.  Same recipe as Tiny (pre-norm layer norm, GELU, d_ff = 4 d, tied output table,
final norm, Chapter 14's starts with seed 1865), four layers each, head width 16, context 128 (the windows have
128 inputs, so no position row goes unused).  Widths 32, 48, 64, 96, 128 make the non-embedding count grow by
about 2.25, 1.78, 2.25, 1.78: roughly doubling.  `python3 sizes.py` prints the counts, read from the tables."""
import _common
from tiny.model import Tiny
WIDTHS = {"s1": 32, "s2": 48, "s3": 64, "s4": 96, "s5": 128, "s6": 192}
FIVE = ["s1", "s2", "s3", "s4", "s5"]      # the five fitted; s6 is held back to test a prediction
LAYERS, HEAD, CONTEXT, VOCAB = 4, 16, 128, 2048
TOKENS_PER_STEP = 8 * 128          # 8 windows of 129 tokens give 1,024 predictions
EMB = ("token table", "position table")

def make(size, seed=1865):
    d = WIDTHS[size]
    return Tiny(vocab_size=VOCAB, n_layers=LAYERS, d_model=d, n_heads=d // HEAD, context=CONTEXT, seed=seed)

def counts(size):
    parts = make(size).count()
    emb = sum(parts[k] for k in EMB)
    total = sum(parts.values())
    return dict(width=WIDTHS[size], parts=parts, embedding=emb, non_embedding=total - emb, total=total)

N = {s: counts(s)["non_embedding"] for s in WIDTHS}

if __name__ == "__main__":
    out = {s: counts(s) for s in WIDTHS}
    prev = None
    for s, c in out.items():
        r = f"  x{c['non_embedding'] / prev:.2f}" if prev else ""
        print(f"{s} width {c['width']:3d}: non-embedding {c['non_embedding']:9,d}  embedding {c['embedding']:9,d}  "
              f"total {c['total']:9,d}  embedding share {c['embedding'] / c['total']:.2f}{r}")
        prev = c["non_embedding"]
    from tiny.model import Tiny as T
    p = T().count(); e = sum(p[k] for k in EMB)
    print(f"Tiny (Chapter 14) for comparison: non-embedding {sum(p.values()) - e:,d}, total {sum(p.values()):,d}")
    _common.save_json("sizes", out)
