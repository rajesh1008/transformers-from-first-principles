"""Chapter 48, the capstone corpus.  Strips the Project Gutenberg header and footer from each downloaded book (corpus/raw/pgID.txt,
fetched from https://www.gutenberg.org/cache/epub/ID/pgID.txt on 12 Nov 2026), takes books in the listed order until the training
text reaches 10,000,000 bytes, and writes corpus/train.txt, corpus/heldout_timemachine.txt and corpus/books.json (title, author,
bytes).  The Time Machine (35) is held out whole.  Every book is public domain in the USA (Project Gutenberg) and its author died
before 1956.  The PG licence text is removed with the header and footer, as PG asks when the trademark is not used."""
import os, re, json
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "corpus", "raw")
ORDER = [1342, 84, 1661, 98, 1400, 768, 1260, 345, 174, 158, 161, 105, 141, 730, 120, 2852, 244, 36, 5230, 2701, 766, 1184, 1399]
HELD = 35
def body(pid):
    t = open(os.path.join(RAW, f"pg{pid}.txt"), encoding="utf-8-sig").read().replace("\r\n", "\n")
    title = re.search(r"^Title:\s*(.+)$", t, re.M); author = re.search(r"^Author:\s*(.+)$", t, re.M)
    s = re.search(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*$", t, re.M); e = re.search(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*$", t, re.M)
    b = t[s.end():e.start()].strip() + "\n"
    return (title.group(1).strip() if title else str(pid)), (author.group(1).strip() if author else "?"), b
books, parts, total = [], [], 0
for pid in ORDER:
    if total >= 10_000_000: break
    ti, au, b = body(pid); n = len(b.encode()); books.append(dict(id=pid, title=ti, author=au, bytes=n)); parts.append(b); total += n
ti, au, held = body(HELD)
open(os.path.join(HERE, "corpus", "train.txt"), "w").write("\n\n".join(parts))
open(os.path.join(HERE, "corpus", "heldout_timemachine.txt"), "w").write(held)
json.dump(dict(train=books, train_bytes=total, heldout=dict(id=HELD, title=ti, author=au, bytes=len(held.encode()))), open(os.path.join(HERE, "corpus", "books.json"), "w"), indent=1)
for b in books: print(f"{b['id']:5d} {b['bytes']:>10,}  {b['title'][:45]:45s} {b['author']}")
print(f"training text: {len(books)} books, {total:,} bytes;  held out: {ti} ({len(held.encode()):,} bytes)")
