"""Chapter 14, section 14.4: why the Logit Lens at height 0 scores so badly in GPT-2 small.  The bare
token-plus-position row, through ln_f and the tied token table: how often is the top guess the seat's
own token, and how much share does the own token get?  Same 1,024 tokens as gpt2_depth.py.
Run from 04-code/ch14:  python3 gpt2_lens0.py   (under a minute)"""
from _common import *
from _gpt2run import weights, norm, softmax
from _bpe import encode
from tiny.corpus import load_text

W = weights()
alice = load_text()
i0 = alice.index("Alice was beginning")
ids = np.array([50256] + encode(alice[i0:i0 + 12000].replace("\r", ""))[:1023])
n = len(ids)
E = W["wte.weight"]
x = E[ids] + W["wpe.weight"][:n]
S = norm(x, W["ln_f.weight"], W["ln_f.bias"]) @ E.T
P = softmax(S)
own = P[np.arange(n), ids]
top = S.argmax(-1)
out = print
if True:
    out(f"GPT-2 small, height 0 through the Logit Lens, {n} seats (seat 0 = end-of-text), float32")
    out(f"  seats whose top guess is their own token: {np.mean(top == ids):.3f}")
    out(f"  average share given to the own token: {own.mean():.3f}; median {np.median(own):.3f}")
    out(f"  seats 1 to {n-1} only: own-token top {np.mean(top[1:] == ids[1:]):.3f}, average own share {own[1:].mean():.3f}")
