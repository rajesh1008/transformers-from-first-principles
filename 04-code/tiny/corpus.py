"""The training corpus: Alice's Adventures in Wonderland (1865), public domain.
Source edition: Project Gutenberg eBook #11 (plain text, UTF-8), file pg11.txt,
'Most recently updated: June 26, 2025', fetched 31 October 2026. The Gutenberg
licence header and footer are removed; the novel's text itself is public domain."""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "alice_gutenberg_raw.txt")

def load_text() -> str:
    """Return the novel's text only, without the Gutenberg header and footer."""
    raw = open(RAW, encoding="utf-8").read()
    start = raw.index("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = raw.index("\n", start) + 1
    end = raw.index("*** END OF THE PROJECT GUTENBERG EBOOK")
    return raw[start:end].strip()

def words(text: str) -> list[str]:
    """Chapter 1's tokenizer: lower-case runs of letters (apostrophes kept inside words).
    Chapter 2 replaces this with a real tokenizer."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())

RAW_HELDOUT = os.path.join(HERE, "..", "data", "looking_glass_gutenberg_raw.txt")

def load_heldout() -> str:
    """Through the Looking-Glass (1871), public domain; Project Gutenberg eBook #12,
    'Most recently updated: June 11, 2026', fetched 31 October 2026. Used only as text
    the tokenizer has never seen, to test how well a vocabulary trained on the novel travels."""
    raw = open(RAW_HELDOUT, encoding="utf-8").read()
    start = raw.index("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = raw.index("\n", start) + 1
    end = raw.index("*** END OF THE PROJECT GUTENBERG EBOOK")
    return raw[start:end].strip()
