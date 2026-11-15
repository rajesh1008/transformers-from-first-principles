"""Chapter 15: the hand-sized Tiny.  Every table is a table the reader has already met:
  token table   Chapter 4's toy table, twelve rows of two numbers (tiny.embed.TOY_EMBED)
  position table Chapter 5's hand-chosen rows for seats 0 to 5 (tiny.embed.TOY_POS)
  attention     Chapter 8's hand head (W_q, W_k, W_v), one head of width 2; W_o is the 2 x 2 identity
  feed-forward  Chapter 12's hand block 2 -> 3 -> 2 with ReLU, b_1 = (0, 0, -2), b_2 = (0, 0)
  norms         RMSNorm with gain (1, 1) (Chapter 13: layer norm on two numbers keeps only a sign)
Both layers use the same tables, so the second layer is the first one again on new rows (a hand
simplification: Tiny's own layers each get their own tables).  The output is tied to the token table.

hand_tiny() builds it from Tiny's own classes:
  Tiny(vocab_size=12, n_layers=2, d_model=2, n_heads=1, d_ff=3, context=6, activation="relu", norm="rms")
and loads the tables by name through Tiny.load, so the shape check of Chapter 14 runs on every table."""
import numpy as np
from tiny.embed import TOY_EMBED, TOY_POS
from tiny.tokenizer import TOY_VOCAB
from tiny.model import Tiny

WORDS = list(TOY_VOCAB)                     # twelve entries, ids 0 to 11
IDS = [0, 1, 2, 3, 0, 4]                    # the running sentence under the toy vocabulary
SEAT_WORDS = [WORDS[i] for i in IDS]        # the, cat, sat, on, the, mat
W_q = np.array([[1, 0], [0, 2]], dtype=float)
W_k = np.array([[0, 1], [1, 0]], dtype=float)
W_v = np.array([[1, 0], [1, 1]], dtype=float)
W_o = np.eye(2)
W_1 = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
b_1 = np.array([0, 0, -2], dtype=float)
W_2 = np.array([[0, 0.5], [0.5, 0], [-0.5, 0.5]], dtype=float)
b_2 = np.zeros(2)
CONFIG = dict(vocab_size=12, n_layers=2, d_model=2, n_heads=1, d_ff=3, context=6, activation="relu", norm="rms")


def layer_tables():
    return {"ln_1.gain": np.ones(2), "att.W_q": W_q, "att.W_k": W_k, "att.W_v": W_v, "att.W_o": W_o,
            "ln_2.gain": np.ones(2), "ff.W_1": W_1, "ff.b_1": b_1, "ff.W_2": W_2, "ff.b_2": b_2}


def hand_tables(E=None, P=None, positions=True):
    t = {"E": np.array(TOY_EMBED if E is None else E, dtype=float),
         "P": np.array(TOY_POS if P is None else P, dtype=float) if positions else np.zeros((6, 2)),
         "ln_f.gain": np.ones(2)}
    for l in range(2):
        for k, v in layer_tables().items():
            t[f"h.{l}.{k}"] = np.array(v, dtype=float)
    return t


def hand_tiny(E=None, P=None, positions=True, **overrides):
    cfg = dict(CONFIG); cfg.update(overrides)
    m = Tiny(**cfg)
    return m.load(hand_tables(E, P, positions))
