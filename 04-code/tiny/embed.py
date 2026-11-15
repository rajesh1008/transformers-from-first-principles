"""Token embeddings (Chapter 4) and position rows (Chapter 5).

Chapter 4: a lookup table with one row of numbers per vocabulary entry.
Chapter 5: a second table with one row per position, or a set of clock hands computed from
the position, added to every token's row.  Chapter 6 adds rotation here.  NumPy only."""
import math
import numpy as np

# ---------------------------------------------------------------- the hand-sized toy table
# Two numbers per entry of Chapter 2's twelve-word toy vocabulary, chosen by hand so that
# every calculation in Chapters 4 to 15 can be done with a pencil.  Rows 0-4 are the
# running sentence's tokens; the others are there so that the table is complete.
TOY_EMBED = np.array([
    [ 1,  0],   #  0 the
    [ 0,  1],   #  1 cat
    [ 1,  1],   #  2 sat
    [ 1, -1],   #  3 on
    [-1,  1],   #  4 mat
    [ 2,  0],   #  5 a
    [ 0,  2],   #  6 dog
    [-1,  2],   #  7 rabbit
    [ 2,  1],   #  8 looked
    [ 2, -1],   #  9 at
    [-2,  1],   # 10 hat
    [ 0,  0],   # 11 <unk>
], dtype=float)

# ---------------------------------------------------------------- arrows (Math Interlude)
def dot(a, b):
    """How much two lists agree: multiply matching entries and add them up."""
    return sum(x * y for x, y in zip(a, b))

def length(a):
    """The length of an arrow: the square root of its dot product with itself (Pythagoras)."""
    return dot(a, a) ** 0.5

def cosine(a, b):
    """Agreement with length taken out: 1 = same direction, 0 = at right angles, -1 = opposite."""
    return dot(a, b) / (length(a) * length(b))

# ---------------------------------------------------------------- Tiny's table
TINY_D = 128            # numbers per token in Tiny (its "width"); fixed in Chapter 4

class Embedding:
    """One row of `d` numbers for each of `vocab_size` tokens.

    The numbers start small and random (spread 0.02, as in GPT-2) and are learned in
    training (Chapters 17 and 24).  The same table is used twice: to turn token ids into
    vectors at the input, and to score every possible next token at the output (tying)."""

    def __init__(self, vocab_size, d=TINY_D, seed=1865, spread=0.02):
        rng = np.random.default_rng(seed)
        self.table = rng.normal(0.0, spread, size=(vocab_size, d))

    @property
    def shape(self):
        return self.table.shape

    def __call__(self, ids):
        """Look up the row for each id: a list of n ids becomes n rows of d numbers."""
        return self.table[np.asarray(ids)]

    def scores(self, h):
        """Tied output: the dot product of the vector h with every row, one score per token."""
        return (self.table * h).sum(axis=1)

    def save(self, path):
        np.save(path, self.table)

    @classmethod
    def load(cls, path):
        e = cls.__new__(cls)
        e.table = np.load(path)
        return e


# ================================================================ Chapter 5: positions
TINY_CONTEXT = 1024     # Tiny's longest sequence: one position row for each of 1,024 places

# Hand-chosen stand-ins for a *learned* position table, two numbers per position, positions 0-5.
# Halves, so that every sum stays pencil-friendly.  Training would find its own numbers.
TOY_POS = np.array([
    [ 0.5,  0.5],   # position 0
    [ 0.5,  0.0],   # position 1
    [ 0.0, -0.5],   # position 2
    [-0.5,  0.0],   # position 3
    [-0.5,  0.5],   # position 4
    [ 0.0,  0.5],   # position 5
])

def add_rows(a, b):
    """Vector adding, one pair of matching entries at a time (Chapter 4's tip-to-tail rule)."""
    return [x + y for x, y in zip(a, b)]

def add_positions(token_rows, position_rows):
    """Row k of the answer = token row k + position row k.  The whole of Chapter 5 in one line."""
    return np.array([add_rows(t, p) for t, p in zip(token_rows, position_rows)])

def clock(pos, degrees_per_step=30.0):
    """A single clock hand that moves `degrees_per_step` for every token, starting at 12 o'clock.
    Returns (across, up) = (sin, cos) of the angle: where the tip of a length-1 hand points."""
    angle = math.radians(pos * degrees_per_step)
    return [math.sin(angle), math.cos(angle)]

def clock_speeds(d=TINY_D, base=10000.0):
    """How far each of the d/2 clocks turns per token, in radians (one full turn = 2*pi = 6.283).
    Clock 0 turns 1 radian (about 57 degrees) per token; each clock after it is slower by the same factor."""
    return [1.0 / base ** (2 * i / d) for i in range(d // 2)]

def sinusoidal(n_positions, d=TINY_D, base=10000.0):
    """The sinusoidal position rows: for each position, the (sin, cos) tip of d/2 clocks of falling speed.
    Nothing here is learned; any position can be computed, including ones no table has a row for."""
    speeds = clock_speeds(d, base)
    rows = []
    for pos in range(n_positions):
        row = []
        for speed in speeds:
            angle = pos * speed
            row += [math.sin(angle), math.cos(angle)]
        rows.append(row)
    return np.array(rows)

class PositionTable:
    """A learned position table: one row of `d` numbers for each of `context` positions.

    It is a second lookup table, exactly like Embedding, except that it is indexed by *where* a
    token stands (0, 1, 2, ...) instead of *which* token it is.  Spread 0.01, as in GPT-2.
    Seed + 1 so that the two tables never share random numbers."""

    def __init__(self, context=TINY_CONTEXT, d=TINY_D, seed=1865, spread=0.01):
        rng = np.random.default_rng(seed + 1)
        self.table = rng.normal(0.0, spread, size=(context, d))

    @property
    def shape(self):
        return self.table.shape

    @property
    def context(self):
        return self.table.shape[0]

    def __call__(self, positions):
        """Look up the row for each position.  A position past the last row is an IndexError."""
        return self.table[np.asarray(positions)]

    def save(self, path):
        np.save(path, self.table)

    @classmethod
    def load(cls, path):
        p = cls.__new__(cls)
        p.table = np.load(path)
        return p

def embed_with_positions(ids, tokens, positions, start=0):
    """Tiny's input block: token row + position row for every token.  `positions` is a PositionTable
    (learned, Tiny's default) or an array of precomputed sinusoidal rows.  `start` shifts the positions."""
    where = list(range(start, start + len(ids)))
    if isinstance(positions, PositionTable):
        P = positions(where)
    else:
        P = positions[where]
    return tokens(ids) + P


# ================================================================ Chapter 6: rotary positions
def rotate_pair(across, up, angle):
    """Turn the arrow (across, up) clockwise by `angle` radians, the way a clock hand turns.
    Two formulas and no table: new across = across*cos + up*sin;  new up = up*cos - across*sin."""
    c, s = math.cos(angle), math.sin(angle)
    return across * c + up * s, up * c - across * s

TOY_TURN = [math.radians(30.0)]     # the toy's single clock: 30 degrees (one hour) per token

def apply_rotary(rows, positions, base=10000.0, speeds=None):
    """Rotary positions.  For each row, turn every pair of numbers (2i, 2i+1) by position x speed_i,
    where speed_i is clock i of Chapter 5.  Lengths never change and nothing is learned.
    In a real model this is done to queries and keys (Chapter 8), never to values."""
    rows = np.asarray(rows, dtype=float)
    if speeds is None:
        speeds = clock_speeds(rows.shape[1], base)
    out = np.zeros_like(rows)
    for r, pos in enumerate(positions):
        for i, speed in enumerate(speeds):
            out[r, 2 * i], out[r, 2 * i + 1] = rotate_pair(rows[r, 2 * i], rows[r, 2 * i + 1], pos * speed)
    return out

def agreement_table(asking, answering=None):
    """Every asking row's dot product with every answering row, by plain loops.
    Row = the token asking, column = the token answering.  Chapter 7 does this in one line."""
    answering = asking if answering is None else answering
    return np.array([[dot(a, b) for b in answering] for a in asking])

class Rotary:
    """Tiny's rotary option: no table, nothing learned (0 numbers).  Chapter 8 calls it on the
    queries and keys; `d` is the width it turns (128 until Chapter 9 splits Tiny into heads).
    Tiny's default stays the learned PositionTable, so that GPT-2's can be loaded in Chapter 23."""

    def __init__(self, d=TINY_D, base=10000.0):
        self.d, self.base = d, base
        self.speeds = clock_speeds(d, base)

    def __call__(self, rows, start=0):
        """Turn row k by the clocks for position start + k."""
        return apply_rotary(rows, range(start, start + len(rows)), speeds=self.speeds)
