#!/usr/bin/env python3
"""Seeded smoke checkpoint for imported Tiny tokenizer."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiny.tokenizer import WordTokenizer, TOY_VOCAB

SENTENCE = "the cat sat on the mat"
tok = WordTokenizer(TOY_VOCAB)
ids = tok.encode(SENTENCE)
assert ids == [0, 1, 2, 3, 0, 4], ids
assert tok.decode(ids) == SENTENCE
print("seeded toy ids:", ids)
print("decoded:", tok.decode(ids))
print("vocab size:", len(TOY_VOCAB))
print("checkpoint ok")
