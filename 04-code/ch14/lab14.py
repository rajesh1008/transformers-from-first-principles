"""Lab 14: Tiny's forward pass, three breakages and a checkpoint.
Part 1  the whole untrained Tiny (12 layers, seed 1865) on the running sentence: scores, shares, top five.
Part 2  Breakage 1: a table loaded in the wrong shape (a) where the shapes clash and (b) where they do not;
        Breakage 2: twelve layers that are one layer ([Block()] * 12);
        Breakage 3: the final norm left out, on untrained Tiny and on GPT-2 small (comparison only).
Part 3  six checkpoint tests.
Run from 04-code/ch14:  python3 lab14.py   (about half a minute; the GPT-2 part reads its weights)"""
from _common import *
import tempfile, hashlib
from tiny.model import Tiny
from tiny.block import Block
from tiny.attention import softmax

t = tok(); ids = sentence_ids(t); pieces = [piece(t, i) for i in ids]
m = Tiny()
S = m.forward(ids); P = softmax(S)
top5 = [piece(t, i) for i in np.argsort(-P[-1])[:5]]
print(f"Part 1. untrained Tiny, seed {SEED}, {m.n_layers} layers, {m.n_params:,} learned numbers")
print(f"  tokens {pieces}")
print(f"  scores {S.shape}; shares {P.shape}; every row sums to one: {bool(np.allclose(P.sum(1), 1))}")
print(f"  top five after '{pieces[-1]}': {top5}  (shares {r3(np.sort(P[-1])[::-1][:5], 6)})")
def fingerprint(model):
    """A short digest of every table, in order: two models with the same digest hold the same numbers."""
    h = hashlib.sha256()
    for k, v in model.tables().items():
        h.update(k.encode()); h.update(np.ascontiguousarray(v, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]
fp = fingerprint(m)
print(f"  fingerprint of all tables: {fp}")
tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "tiny_ch14.npz")
m.save(path)
m2 = Tiny(seed=7).load(path)   # a different seed, so only the file can make them match
print(f"  saved and loaded back: same scores {bool(np.array_equal(m2.forward(ids), S))}, same fingerprint {fingerprint(m2) == fp}")

print("\nBreakage 1. a table loaded in the wrong shape")
d = dict(np.load(path))
d1 = dict(d); d1["h.3.ff.W_1"] = d["h.3.ff.W_1"].T          # (128, 512) stored as (512, 128)
try:
    Tiny().load(d1)
except ValueError as e:
    print(f"  (a) with the shape check: refused: {e}")
mb = Tiny().load(d1, check_shapes=False)
try:
    mb.forward(ids)
except ValueError as e:
    print(f"  (a) without the shape check it loads, then the forward pass fails: {str(e).splitlines()[0][:110]}")
d2 = dict(d); d2["h.3.att.W_o"] = d["h.3.att.W_o"].T        # (128, 128): the transpose has the same shape
mt = Tiny().load(d2)
S2 = mt.forward(ids); P2 = softmax(S2)
print(f"  (b) layer 3's attention output table transposed (128 x 128, so the shape check passes): loads and runs")
print(f"      top five after '{pieces[-1]}': {[piece(t, i) for i in np.argsort(-P2[-1])[:5]]}")
print(f"      largest change in any score {np.abs(S2 - S).max():.4f}; ' sat''s top row moves by {length(mt.top_rows(mt.forward(ids, keep=True)[1][-1])[2] - m.top_rows(m.forward(ids, keep=True)[1][-1])[2]):.4f}")
print(f"      fingerprint {fingerprint(mt)} against {fp}: the test that catches it")

print("\nBreakage 2. twelve layers that are one layer")
mc = Tiny(); mc.blocks = [Block(layer=0)] * 12
naive = sum(b.n_params for b in mc.blocks)
print(f"  learned numbers by adding up the layers: {naive + 393216 + 256:,}; counted from unique tables: {mc.n_params:,}")
print(f"  distinct layer objects: {len(set(map(id, mc.blocks)))}; distinct attention query tables: {len({id(b.att.W_q) for b in mc.blocks})}")
Sc = mc.forward(ids)
print(f"  it runs: scores {Sc.shape}, top five {[piece(t, i) for i in np.argsort(-Sc[-1])[:5]]}")
from tiny.norm import token_similarity
_, s_ok = m.forward(ids, keep=True); _, s_bad = mc.forward(ids, keep=True)
print(f"  token similarity at the top: {token_similarity(s_ok[-1]):.3f} (twelve layers) against {token_similarity(s_bad[-1]):.3f} (one layer twelve times)")

print("\nBreakage 3. the final norm left out")
m0 = Tiny(final_norm=False)
ids_n = novel_ids(1025, t)
print(f"  untrained Tiny: top row length {length(s_ok[-1][-1]):.3f} reaches the scores instead of {length(m.top_rows(s_ok[-1])[-1]):.3f}; "
      f"loss on the novel {m0.loss(ids_n):.4f} against {m.loss(ids_n):.4f}: nothing visible")
from _gpt2run import weights, block, norm, L as GL
from _bpe import encode
from tiny.corpus import load_text
W = weights(); alice = load_text(); i0 = alice.index("Alice was beginning")
gid = np.array([50256] + encode(alice[i0:i0 + 12000].replace("\r", ""))[:1023]); n = len(gid)
x = W["wte.weight"][gid] + W["wpe.weight"][:n]
for l in range(GL):
    x, _ = block(x, l)
def gloss(Z):
    S_ = Z[:-1] @ W["wte.weight"].T; S_ = S_ - S_.max(-1, keepdims=True)
    lp = S_ - np.log(np.exp(S_).sum(-1, keepdims=True)); return float(-lp[np.arange(n - 1), gid[1:]].mean())
with_f = gloss(norm(x, W["ln_f.weight"], W["ln_f.bias"])); without_f = gloss(x)
print(f"  GPT-2 small on the same 1,024 tokens of the novel: loss {with_f:.3f} with ln_f, {without_f:.3f} without")

print("\nPart 3. checkpoint tests")
tests = [
    ("scores are (7, 2048) and every row of shares sums to one", S.shape == (7, 2048) and np.allclose(P.sum(1), 1)),
    ("the top five after 'at' are " + str(top5), top5 == ["hesh", " squ", "<0xD8>", "at", "not"]),
    ("Tiny holds 2,766,592 learned numbers, counted from unique tables (catches Breakage 2)", m.n_params == 2766592 and mc.n_params != 2766592),
]
def refused(dd):
    try:
        Tiny().load(dd); return False
    except ValueError:
        return True
tests.append(("the shape check refuses a transposed 128 x 512 table (Breakage 1a)", refused(d1)))
tests.append(("the fingerprint differs when a square table is transposed (Breakage 1b)", fingerprint(mt) != fp and fingerprint(m2) == fp))
tests.append(("the top rows reaching the scores have length close to sqrt(128) = 11.31 (catches Breakage 3)",
              abs(np.linalg.norm(m.top_rows(s_ok[-1]), axis=1).mean() - np.sqrt(128)) < 0.1))
for i, (name, ok) in enumerate(tests, 1):
    print(f"  {i}. {name}: {'pass' if ok else 'FAIL'}")
print("all checkpoint tests pass:", all(ok for _, ok in tests))
