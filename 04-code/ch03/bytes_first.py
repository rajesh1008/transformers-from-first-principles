"""Bytes first: every text is already a list of numbers from 0 to 255.
Run from 04-code/ch03:  python3 bytes_first.py"""
import _common
from _common import SENTENCE
from tiny.corpus import load_text

b = SENTENCE.encode("utf-8")                 # text -> bytes (UTF-8)
print(f"{SENTENCE!r}: {len(SENTENCE)} characters, {len(b)} bytes")
print("bytes:", list(b))
print("back again:", bytes(b).decode("utf-8"))

for text in ["Où est ma chatte?", "“", "naïve café"]:
    bb = text.encode("utf-8")
    print(f"\n{text!r}: {len(text)} characters, {len(bb)} bytes")
    for ch in text:
        if len(ch.encode("utf-8")) > 1:
            print(f"  {ch!r} is {len(ch.encode('utf-8'))} bytes: {list(ch.encode('utf-8'))}")

novel = load_text().encode("utf-8")
used = sorted(set(novel))
print(f"\nthe novel: {len(novel):,} bytes, using {len(used)} of the 256 possible byte values")
print("byte values above 127 that occur:", [v for v in used if v > 127])
