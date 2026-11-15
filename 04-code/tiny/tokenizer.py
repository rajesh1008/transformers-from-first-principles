"""Tiny's tokenizer.  Chapter 2: a character tokenizer and a toy word vocabulary.
Chapter 3: byte-level byte-pair encoding (train_bpe, BPETokenizer)."""
import re, heapq
from collections import Counter, defaultdict

def split_words(text):
    """Lower-case words (apostrophes kept inside), and every other mark on its own."""
    text = text.replace("\u2019", "'").replace("\u2018", "'").lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[^\sa-z]", text)

class WordTokenizer:
    """A fixed list of words. Position in the list is the id."""
    def __init__(self, vocab, unk="<unk>"):
        self.itos = list(vocab)                       # id -> word
        self.stoi = {w: i for i, w in enumerate(vocab)}   # word -> id
        self.unk = unk

    def encode(self, text, strict=False):
        ids = []
        for w in split_words(text):
            if strict or w in self.stoi:
                ids.append(self.stoi[w])              # strict: unknown words raise KeyError
            else:
                ids.append(self.stoi[self.unk])
        return ids

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids)

class CharTokenizer:
    """Every distinct character in the training text gets an id."""
    def __init__(self, text):
        self.itos = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

TOY_VOCAB = ["the", "cat", "sat", "on", "mat", "a", "dog",
             "rabbit", "looked", "at", "hat", "<unk>"]


# ---------------------------------------------------------------- Chapter 3: byte-pair encoding

# Cut text into chunks before merging, so that no merge ever crosses a chunk boundary.
# A chunk is: an English contraction ending, or an optional space followed by letters,
# by digits, or by other marks; or a run of white space.  (Close to GPT-2's own rule.)
CHUNK = re.compile(r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")

def chunks(text):
    return CHUNK.findall(text)

def train_bpe(text, vocab_size, split=chunks, min_count=2):
    """Learn merges until the vocabulary has vocab_size entries, or no pair occurs
    min_count times.  Returns the merges in order: a list of ((a, b), new_id, count).
    Ties are broken towards the pair with the smaller first id, then smaller second id."""
    counts = Counter(split(text))
    words = [list(c.encode("utf-8")) for c in counts]      # each chunk as a list of byte ids
    freq = list(counts.values())
    pairs, where = Counter(), defaultdict(set)
    for wi, w in enumerate(words):
        for p in zip(w, w[1:]):
            pairs[p] += freq[wi]
            where[p].add(wi)
    heap = [(-n, p) for p, n in pairs.items()]
    heapq.heapify(heap)
    merges, next_id = [], 256
    while next_id < vocab_size and heap:
        n, best = heapq.heappop(heap)
        if -n != pairs.get(best, 0):                       # stale entry: the count has changed
            continue
        if -n < min_count:
            break
        merges.append((best, next_id, -n))
        changed = set()
        for wi in where.pop(best):
            w = words[wi]
            for p in zip(w, w[1:]):                       # take this chunk's pairs out ...
                pairs[p] -= freq[wi]
                changed.add(p)
            w = merge_pair(w, best, next_id)
            words[wi] = w
            for p in zip(w, w[1:]):                       # ... and put the new ones back
                pairs[p] += freq[wi]
                where[p].add(wi)
                changed.add(p)
        del pairs[best]
        for p in changed - {best}:                        # every changed count goes back on the heap
            heapq.heappush(heap, (-pairs[p], p))
        next_id += 1
    return merges

def merge_pair(ids, pair, new_id):
    """Replace every occurrence of pair (left to right, no overlaps) with new_id."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

class BPETokenizer:
    """Byte-level BPE.  Ids 0-255 are the bytes; each merge adds one id; special tokens come last."""
    def __init__(self, merges, specials=("<|endoftext|>",)):
        self.merges = list(merges)
        self.ranks = {pair: new_id for pair, new_id, _ in merges}   # pair -> id (lower = learned earlier)
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id, _ in merges:
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self.specials = {s: len(self.vocab) + k for k, s in enumerate(specials)}
        for s, i in self.specials.items():
            self.vocab[i] = s.encode("utf-8")
        self.cache = {}

    def __len__(self):
        return len(self.vocab)

    def encode_chunk(self, chunk):
        if chunk in self.cache:
            return self.cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break                                     # no learned merge applies any more
            ids = merge_pair(ids, pair, self.ranks[pair])
        self.cache[chunk] = ids
        return ids

    def encode(self, text, allowed_special=False):
        if allowed_special and self.specials:
            pattern = "(" + "|".join(re.escape(s) for s in self.specials) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]
        ids = []
        for part in parts:
            if part in self.specials and allowed_special:
                ids.append(self.specials[part])
            else:
                for c in chunks(part):
                    ids.extend(self.encode_chunk(c))
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def piece(self, i):
        """The text of one token, with spaces shown as a visible mark."""
        return self.vocab[i].decode("utf-8", errors="replace").replace(" ", "\u2423")

    def save(self, path):
        """One merge per line: left id, right id, new id, and how often the pair was seen."""
        with open(path, "w") as f:
            for (a, b), new_id, n in self.merges:
                f.write(f"{a} {b} {new_id} {n}\n")

def load_bpe(path):
    merges = []
    for line in open(path):
        a, b, i, n = map(int, line.split())
        merges.append(((a, b), i, n))
    return BPETokenizer(merges)

TINY_VOCAB_SIZE = 2048      # Tiny's vocabulary: 256 bytes + 1,791 merges + 1 special token
