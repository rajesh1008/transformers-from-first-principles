"""The merge table is the vocabulary: every id is a byte, or two earlier ids glued together.
Run from 04-code/ch03:  python3 merge_table.py"""
import _common
from collections import Counter
from tiny.corpus import load_text
from tiny.tokenizer import train_bpe, BPETokenizer, TINY_VOCAB_SIZE

text = load_text()
merges = train_bpe(text, TINY_VOCAB_SIZE - 1)
tok = BPETokenizer(merges)
parent = {new_id: pair for pair, new_id, _ in merges}

def tree(i, depth=0):                       # how an id was built, one line per step
    line = "    " * depth + f"{i:>4}  {tok.piece(i)!r}"
    if i in parent:
        a, b = parent[i]
        return [line] + tree(a, depth + 1) + tree(b, depth + 1)
    return [line + "  (a byte)"]

for word in [" the", " cat", " looked"]:
    i = tok.encode(word)
    print(f"{word!r} -> {i}")
    print("\n".join(tree(i[0])) + "\n")

print("ids 0-3:", [tok.vocab[i] for i in range(4)], " id 97:", tok.vocab[97], " id 32:", tok.vocab[32])
print("ids 256-261:", [tok.piece(i) for i in range(256, 262)])
print(f"last merge id {merges[-1][1]}: {tok.piece(merges[-1][1])!r};  special token id "
      f"{tok.specials['<|endoftext|>']}")

longest = sorted(range(256, merges[-1][1] + 1), key=lambda i: -len(tok.vocab[i]))[:8]
print("longest pieces:", [tok.piece(i) for i in longest])
whole = sum(1 for i in range(256, merges[-1][1] + 1) if tok.vocab[i].startswith(b" "))
print(f"merged pieces that begin with a space: {whole:,} of {len(merges):,}")

# How often is each merged piece actually used once the whole novel is encoded?
used = Counter(tok.encode(text))
made = {new_id: n for _, new_id, n in merges}
never = [i for i in made if used[i] == 0]
rare = [i for i in made if 0 < used[i] < 0.1 * made[i]]
print(f"\nencoding the novel with the finished table uses {len(used):,} different ids")
print(f"merged pieces never used in the final encoding: {len(never)} ({len(never)/len(merges):.1%})")
print("  for example:", [tok.piece(i) for i in sorted(never, key=lambda i: -made[i])[:8]])
print(f"used, but less than a tenth as often as when they were made: {len(rare)}")
for i in sorted(never, key=lambda i: -made[i])[:4]:
    print(f"  {tok.piece(i)!r:<9} made from a pair seen {made[i]:>3} times; used 0 times in the end")
