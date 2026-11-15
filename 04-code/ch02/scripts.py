"""The running sentence in five languages and four scripts, counted.
Translations by the author; tiktoken is used for comparison only.
Run from 04-code/ch02:  python3 scripts.py"""
import _common
import tiktoken

SENTENCES = [("English", "Latin", "the cat sat on the mat"),
             ("German", "Latin", "die Katze saß auf der Matte"),
             ("Greek", "Greek", "η γάτα κάθισε στο χαλάκι"),
             ("Hindi", "Devanagari", "बिल्ली चटाई पर बैठी"),
             ("Japanese", "Japanese", "猫はマットの上に座った")]
gpt2, o200k = tiktoken.get_encoding("gpt2"), tiktoken.get_encoding("o200k_base")
base = {}
print(f"{'language':<9} {'script':<11} {'chars':>5} {'bytes':>5} {'GPT-2':>6} {'o200k':>6}   GPT-2 vs English   o200k vs English")
for lang, script, s in SENTENCES:
    n2, n4 = len(gpt2.encode(s)), len(o200k.encode(s))
    if lang == "English": base = {"g": n2, "o": n4}
    print(f"{lang:<9} {script:<11} {len(s):>5} {len(s.encode('utf-8')):>5} {n2:>6} {n4:>6}"
          f"   {n2/base['g']:>6.1f}x           {n4/base['o']:>6.1f}x")
print()
for ch in ["a", "ß", "γ", "ब", "猫"]:
    print(f"{ch!r}: {len(ch.encode('utf-8'))} byte(s) in UTF-8: {list(ch.encode('utf-8'))}")
