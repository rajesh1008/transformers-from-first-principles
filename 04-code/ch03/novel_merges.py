"""Train byte-level BPE on the novel and look at the merges it learns, in order.
Run from 04-code/ch03:  python3 novel_merges.py   (tiktoken is used for comparison only)"""
import _common, time
import tiktoken
from tiny.corpus import load_text
from tiny.tokenizer import train_bpe, BPETokenizer, chunks, TINY_VOCAB_SIZE

text = load_text()
cs = chunks(text)
print(f"the novel cut into {len(cs):,} chunks, {len(set(cs)):,} of them different")
print("the first twelve chunks:", cs[:12])

t0 = time.perf_counter()
merges = train_bpe(text, TINY_VOCAB_SIZE - 1)       # leave one place for the end-of-text token
dt = time.perf_counter() - t0
tok = BPETokenizer(merges)
print(f"\nlearned {len(merges):,} merges in {dt:.2f} s; vocabulary size {len(tok):,} "
      f"(256 bytes + {len(merges):,} merges + 1 special)")

gpt2 = tiktoken.get_encoding("gpt2")
print("\n  #  new piece    made from        seen   id  | GPT-2 id of the same piece")
for k, ((a, b), new_id, n) in enumerate(merges[:24], 1):
    s = tok.vocab[new_id]
    try:
        g = gpt2.encode_single_token(s)
    except KeyError:
        g = "-"
    print(f"{k:>3}  {tok.piece(new_id)!r:<11} {tok.piece(a)!r:>6} + {tok.piece(b)!r:<7} {n:>5}  {new_id:>4}  | {g}")

counts = [n for _, _, n in merges]
for limit in [1000, 100, 10]:
    k = next(i for i, n in enumerate(counts, 1) if n < limit)
    print(f"the best pair is first seen fewer than {limit:,} times at merge {k:,}")
print(f"merge {len(merges):,}, the last one Tiny keeps, joins a pair seen {counts[-1]} times")
full = train_bpe(text, 10**6)
print(f"with no size limit, training stops after {len(full):,} merges, "
      f"when no pair is left that occurs twice")
