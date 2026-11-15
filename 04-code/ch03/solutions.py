"""Computed answers to the Chapter 3 exercises (Appendix A).  Run from 04-code/ch03:  python3 solutions.py"""
import _common, heapq
from collections import Counter
from _common import SENTENCE
from tiny.corpus import load_text, load_heldout
from tiny.tokenizer import train_bpe, BPETokenizer, chunks, merge_pair, TINY_VOCAB_SIZE

text, sequel = load_text(), load_heldout()
TWENTY = "the cat sat on the mat the rat sat on the hat the cat ate the rat that sat on"

m = train_bpe(TWENTY, 256 + 7, split=str.split, min_count=1)
t = BPETokenizer(m, specials=())
print("Ex1 merges 6 and 7:", [(t.piece(a) + "+" + t.piece(b), n) for (a, b), _, n in m[5:7]])
t5 = BPETokenizer(m[:5], specials=())
print("Ex2 'that rat ate' after five merges:", [[t5.piece(i) for i in t5.encode_chunk(w)] for w in "that rat ate".split()])
for s in ["naïve café", "Où est ma chatte?"]:
    print(f"Ex3 {s!r}: {len(s)} characters, {len(s.encode('utf-8'))} bytes")
print(f"Ex4 bytes per token {151094/44360:.2f}; pairs ratio for 10 merges {(127316/44360)**2:.1f}")
first3 = train_bpe(text, 256 + 3)
tok3 = BPETokenizer(first3, specials=())
pred = len(text.encode()) - sum(n for _, _, n in first3)
print(f"Ex5 predicted after 3 merges: {pred:,}; actual: {len(tok3.encode(text)):,}")

tiny = BPETokenizer(train_bpe(text, TINY_VOCAB_SIZE - 1))
top = Counter(tiny.encode(text)).most_common(20)
print("Ex6 twenty most used:", [tiny.piece(i) for i, _ in top])
print("    of which begin with a space:", sum(1 for i, _ in top if tiny.vocab[i].startswith(b" ")))
big = BPETokenizer(train_bpe(text, 50000 - 1, min_count=1))
print(f"Ex7 merges with min_count=1: {len(big) - 257:,}; novel tokens {len(big.encode(text)):,}; chunks {len(chunks(text)):,}")
other = BPETokenizer(train_bpe(sequel, TINY_VOCAB_SIZE - 1))
print(f"Ex8 trained on the sequel: sequel {len(other.encode(sequel)):,} tokens, novel {len(other.encode(text)):,} tokens; "
      f"sentence {[other.piece(i) for i in other.encode(SENTENCE)]}")

def train_big_ids_first(text, vocab_size):         # same as train_bpe, but ties go to the LARGER ids
    counts = Counter(chunks(text)); words = [list(c.encode()) for c in counts]; freq = list(counts.values())
    merges, nid = [], 256
    while nid < vocab_size:
        pc = Counter()
        for wi, w in enumerate(words):
            for p in zip(w, w[1:]): pc[p] += freq[wi]
        if not pc: break
        best, n = max(pc.items(), key=lambda kv: (kv[1], kv[0]))
        if n < 2: break
        merges.append((best, nid, n)); words = [merge_pair(w, best, nid) for w in words]; nid += 1
    return merges
alt = BPETokenizer(train_big_ids_first(text, 600))
ref = BPETokenizer(train_bpe(text, 600))
diff = len({alt.vocab[i] for i in range(256, 600)} ^ {ref.vocab[i] for i in range(256, 600)}) // 2
print(f"Ex9 344 merges, ties to larger ids: {diff} pieces differ between the two vocabularies; sentence "
      f"{[alt.piece(i) for i in alt.encode(SENTENCE)]} vs {[ref.piece(i) for i in ref.encode(SENTENCE)]}; "
      f"novel {len(alt.encode(text)):,} vs {len(ref.encode(text)):,}")
de = "die Katze saß auf der Matte"
print(f"Ex10 German before: {len(tiny.encode(de))} tokens")
mix = BPETokenizer(train_bpe(text + ("\n" + de) * 100, TINY_VOCAB_SIZE - 1))
print(f"     trained with the German sentence 100 more times: {len(mix.encode(de))} tokens "
      f"{[mix.piece(i) for i in mix.encode(de)]}; English sentence {len(mix.encode(SENTENCE))} tokens")
