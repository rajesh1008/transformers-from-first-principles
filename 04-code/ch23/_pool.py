"""Chapter 23 reuses Chapter 22's Carroll pool and held-out sets unchanged (loaded from ../ch22/_pool.py), and
writes the pool once as uint16 shards (one document per book, about 65,536 tokens a shard) for ShardStream."""
import _common
import os, importlib.util
_spec = importlib.util.spec_from_file_location("_pool22", os.path.join(_common.HERE, "..", "ch22", "_pool.py"))
_p22 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_p22)
corpora, pool_windows, heldout_sets, POOL_BOOKS = _p22.corpora, _p22.pool_windows, _p22.heldout_sets, _p22.POOL_BOOKS
from tiny import data as D
import numpy as np
SHARDS = os.path.join(_common.HERE, "outputs", "shards")
PIECE = 4096          # tokens per document piece

def pieces():
    """The pool cut into pieces of 4,096 tokens (the last piece of each book shorter), in a random order (seed 1865).
    The first version of this chapter's ladder wrote each whole book as one document, which gave four shards; with a
    shuffle buffer of 256 windows the stream then read the books one after another, and the book read last decided
    much of the final loss (RUN.txt, outputs/v1_four_shards/).  Cutting and shuffling before writing is Chapter 22's
    advice ("shuffle documents before writing shards")."""
    c = corpora()
    out = []
    for b in POOL_BOOKS:
        t = c["books"][b]
        out += [t[i:i + PIECE] for i in range(0, len(t), PIECE)]
    order = np.random.default_rng(1865).permutation(len(out))
    return [out[i] for i in order]

def shards():
    """Write the shards if they are not there yet (run prep.py once before starting parallel runs)."""
    if not os.path.exists(os.path.join(SHARDS, "index.json")):
        D.write_shards(pieces(), SHARDS, shard_tokens=16384, name="carroll")
    return SHARDS
