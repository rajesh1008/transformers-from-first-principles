"""Encoding replays the merges in the order they were learned; decoding glues bytes back together.
Run from 04-code/ch03:  python3 encode_decode.py"""
import _common
from _common import SENTENCE
from tiny.corpus import load_text
from tiny.tokenizer import train_bpe, BPETokenizer, merge_pair, chunks, TINY_VOCAB_SIZE

text = load_text()
tok = BPETokenizer(train_bpe(text, TINY_VOCAB_SIZE - 1))

def encode_verbose(chunk):
    ids = list(chunk.encode("utf-8"))
    print(f"  start  {[tok.piece(i) for i in ids]}")
    while len(ids) > 1:
        pair = min(zip(ids, ids[1:]), key=lambda p: tok.ranks.get(p, float("inf")))
        if pair not in tok.ranks:
            break
        ids = merge_pair(ids, pair, tok.ranks[pair])
        print(f"  apply merge {tok.ranks[pair] - 255:>4} ({tok.piece(pair[0])!r} + {tok.piece(pair[1])!r})"
              f" -> {[tok.piece(i) for i in ids]}")
    return ids

print("chunks of the sentence:", chunks(SENTENCE))
for c in [" cat", " mat"]:
    print(f"\nencoding {c!r}:")
    print("  result", encode_verbose(c))

ids = tok.encode(SENTENCE)
print(f"\n{SENTENCE!r} -> {ids}")
print("pieces:", [tok.piece(i) for i in ids])
print("decoded:", repr(tok.decode(ids)))

ids = tok.encode(text)
print(f"\nthe whole novel: {len(ids):,} ids; decodes back exactly: {tok.decode(ids) == text}")

for s in ["“Curiouser", "1865–1871"]:
    q = tok.encode(s)
    print(f"\n{s!r} -> {q} = {[tok.piece(i) for i in q]}")
q = tok.encode("–")                                   # an en dash: three bytes, never seen in the novel
print("the dash alone:", q, "bytes", [list(tok.vocab[i]) for i in q])
print("decoding only its first id:", repr(tok.decode(q[:1])), "  decoding both:", repr(tok.decode(q)))

eot = "<|endoftext|>"
print(f"\n{eot!r} typed as text   ->", len(tok.encode(eot)), "ids")
print(f"{eot!r} allowed special ->", tok.encode(eot, allowed_special=True))
