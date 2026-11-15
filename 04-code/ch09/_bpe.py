"""GPT-2's byte-level BPE encoder, written out so that Chapter 9 needs nothing but NumPy and the two
public files vocab.json and merges.txt (Hugging Face, openai-community/gpt2), cached in
~/.cache/tfp/gpt2.  tiktoken is not installed on the book's machine; Chapter 3 compares with it."""
import os, json, urllib.request
import regex
DIR = os.path.expanduser("~/.cache/tfp/gpt2")
BASE = "https://huggingface.co/openai-community/gpt2/resolve/main/"
for f in ("vocab.json", "merges.txt"):
    if not os.path.exists(os.path.join(DIR, f)):
        os.makedirs(DIR, exist_ok=True); urllib.request.urlretrieve(BASE + f, os.path.join(DIR, f))
ENC = json.load(open(os.path.join(DIR, "vocab.json"), encoding="utf-8"))
MERGES = [tuple(l.split()) for l in open(os.path.join(DIR, "merges.txt"), encoding="utf-8").read().split("\n")[1:] if l]
RANK = {m: i for i, m in enumerate(MERGES)}
PAT = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))
B2U = _bytes_to_unicode()

def _bpe(word):
    parts = list(word)
    while len(parts) > 1:
        pairs = [(RANK.get((a, b), 1e18), k) for k, (a, b) in enumerate(zip(parts, parts[1:]))]
        r, k = min(pairs)
        if r == 1e18:
            break
        parts = parts[:k] + [parts[k] + parts[k + 1]] + parts[k + 2:]
    return parts

def encode(text):
    ids = []
    for chunk in PAT.findall(text):
        word = "".join(B2U[b] for b in chunk.encode("utf-8"))
        ids += [ENC[p] for p in _bpe(word)]
    return ids
