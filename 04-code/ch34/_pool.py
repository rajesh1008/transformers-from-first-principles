"""Chapter 24 reads the Carroll pool exactly as Chapter 23 did: the 24 shards of shuffled 4,096-token pieces that
../ch23/prep.py wrote to ../ch23/outputs/shards, and Chapter 22's two held-out sets (Carroll held-out, Looking-Glass)."""
import _common
import os, importlib.util
_spec = importlib.util.spec_from_file_location("_pool23", os.path.join(_common.HERE, "..", "ch23", "_pool.py"))
_p23 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_p23)
heldout_sets, corpora, POOL_BOOKS = _p23.heldout_sets, _p23.corpora, _p23.POOL_BOOKS
SHARDS = os.path.join(_common.HERE, "..", "ch23", "outputs", "shards")
WINDOWS_PER_PASS = 3055           # windows of 129 in one pass of the 24 shards (Chapter 23, prep.py)

def shards():
    assert os.path.exists(os.path.join(SHARDS, "index.json")), "run ../ch23/prep.py once first"
    return SHARDS
