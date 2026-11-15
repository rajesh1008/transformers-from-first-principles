"""Tiny's vocabulary against GPT-2's (and a newer one).  tiktoken is used for comparison only.
Run from 04-code/ch03:  python3 compare_tiktoken.py"""
import _common
import tiktoken
from tiny.corpus import load_text, load_heldout
from tiny.tokenizer import train_bpe, BPETokenizer, TINY_VOCAB_SIZE

text, sequel = load_text(), load_heldout()
tiny = BPETokenizer(train_bpe(text, TINY_VOCAB_SIZE - 1))
gpt2, o200k = tiktoken.get_encoding("gpt2"), tiktoken.get_encoding("o200k_base")
show = lambda enc, ids: "|".join(enc.decode([i]).replace(" ", "\u2423") for i in ids)

print(f"{'':<8}{'vocabulary':>11} {'novel':>8} {'sequel':>8}")
for name, enc, n in [("Tiny", tiny, len(tiny)), ("GPT-2", gpt2, gpt2.n_vocab), ("o200k", o200k, o200k.n_vocab)]:
    print(f"{name:<8}{n:>11,} {len(enc.encode(text)):>8,} {len(enc.encode(sequel)):>8,}")

shared = sum(1 for i in range(256, len(tiny) - 1) if tiny.vocab[i] in gpt2._mergeable_ranks)
print(f"\nof Tiny's {len(tiny) - 257:,} merged pieces, {shared:,} ({shared/(len(tiny)-257):.0%}) are also GPT-2 tokens")

print(f"\n{'text':<16} {'Tiny':<30} {'GPT-2':<26}")
for w in [" Alice", " Rabbit", " Gryphon", " Caterpillar", " mat", " Looking", " computer",
          " strawberry", " 1865", " 2026", " don’t", " Toronto"]:
    mine = "|".join(tiny.piece(i) for i in tiny.encode(w))
    print(f"{w!r:<16} {mine:<30} {show(gpt2, gpt2.encode(w)):<26}")

SENTENCES = [("English", "the cat sat on the mat"), ("German", "die Katze saß auf der Matte"),
             ("Greek", "η γάτα κάθισε στο χαλάκι"), ("Hindi", "बिल्ली चटाई पर बैठी"),
             ("Japanese", "猫はマットの上に座った")]
print(f"\n{'language':<9} {'bytes':>5} | {'Tiny':>4} single-byte | {'GPT-2':>5} single-byte")
for lang, s in SENTENCES:
    t, g = tiny.encode(s), gpt2.encode(s)
    st = sum(1 for i in t if i < 256)
    sg = sum(1 for i in g if len(gpt2.decode_single_token_bytes(i)) == 1)
    print(f"{lang:<9} {len(s.encode('utf-8')):>5} | {len(t):>4} {st:>11} | {len(g):>5} {sg:>11}")
