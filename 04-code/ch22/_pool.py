"""Chapter 22's corpora, built the same way by every script.  The "Carroll pool" is five other public-domain books
by the same author (Sylvie and Bruno, Sylvie and Bruno Concluded, A Tangled Tale, Phantasmagoria, The Hunting of the
Snark), each encoded whole with Tiny's 2,048-token BPE and laid end to end in that order.  The last 4,000 tokens of
Sylvie and Bruno Concluded's novel text are held out (they are never in the pool).  Alice is the novel of Chapters 1 to 21 and
Through the Looking-Glass the held-out book, as before."""
import _common
import numpy as np
from tiny import data as D
from _common import tok, SEED
POOL_BOOKS = ["sylvie", "sylvie2", "tangled", "phantasmagoria", "snark"]
D.SOURCES.setdefault("phantasmagoria", ("phantasmagoria_gutenberg_raw.txt", "Phantasmagoria and Other Poems (1869), eBook #651"))
D.SOURCES.setdefault("snark", ("hunting_of_the_snark_gutenberg_raw.txt", "The Hunting of the Snark (1876), eBook #13"))
HOLD = 4000

def book_text(b):
    """A pool book's text.  Sylvie and Bruno Concluded keeps its novel only: its back matter (a general index, the
    publisher's advertisements and the transcriber's notes, after the line "THE END.") is cut, so that the held-out
    tokens are the novel's last pages rather than advertisements."""
    t = D.load_source(b)
    if b == "sylvie2":
        t = t[:t.index("THE END.")]
    return t
_cache = {}
def corpora():
    if _cache:
        return _cache
    T = tok()
    books = {b: np.array(T.encode(book_text(b)), dtype=np.int64) for b in POOL_BOOKS}
    held_b = books["sylvie2"][-HOLD:]
    books["sylvie2"] = books["sylvie2"][:-HOLD]
    pool = np.concatenate([books[b] for b in POOL_BOOKS])
    book_of = np.concatenate([np.full(len(books[b]), i) for i, b in enumerate(POOL_BOOKS)])
    alice = np.array(T.encode(D.load_source("alice")), dtype=np.int64)
    glass = np.array(T.encode(D.load_source("glass")), dtype=np.int64)
    _cache.update(T=T, books=books, pool=pool, book_of=book_of, held_b=held_b, alice=alice, glass=glass)
    return _cache

def pool_windows():
    c = corpora()
    w, d = D.pack([c["pool"]], 129)
    return w, c["book_of"][np.arange(len(w))[:, None] * 128 + np.arange(129)[None, :]]

def heldout_sets():
    """Looking-Glass: the same 16 windows as Chapters 19 to 21 (seed 1865, drawn after the 400 x 8 training starts
    over the 44,360-token novel).  Carroll held-out: 16 windows at random from the 4,000 held-out tokens."""
    c = corpora()
    rng = np.random.default_rng(SEED)
    rng.integers(0, len(c["alice"]) - 129, size=(400, 8))
    hs = rng.integers(0, len(c["glass"]) - 129, size=16)
    G = np.stack([c["glass"][a:a + 129] for a in hs])
    r2 = np.random.default_rng(SEED + 1)
    hb = r2.integers(0, len(c["held_b"]) - 129, size=16)
    B = np.stack([c["held_b"][a:a + 129] for a in hb])
    return G, B
