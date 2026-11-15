"""The twelve-word toy vocabulary and the running sentence.
Run from 04-code/ch02:  python3 toy_vocab.py"""
import _common
from _common import SENTENCE, prompt
from tiny.tokenizer import WordTokenizer, TOY_VOCAB

tok = WordTokenizer(TOY_VOCAB)
for i, w in enumerate(tok.itos):
    print(f"{i:>2}  {w}")
ids = tok.encode(SENTENCE)
print(f"\n{SENTENCE!r} -> {ids}")
print(f"decoded back            -> {tok.decode(ids)!r}")
print(f"{prompt!r} -> {tok.encode(prompt)}")
print(f'"the cats sat on the mat" -> {tok.encode("the cats sat on the mat")}'
      f"  -> {tok.decode(tok.encode('the cats sat on the mat'))!r}")
