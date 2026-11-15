"""Check (and, if missing, download) the public-domain Project Gutenberg texts the book reads.

The exact editions used for the book (downloaded 10 November 2026) are already in
this folder. Run this from the repository's top folder to confirm they are intact:

    python3 04-code/data/fetch_data.py

A file that is missing is downloaded from Project Gutenberg. Gutenberg corrects its
texts from time to time (A Tangled Tale had already changed by 15 November 2026),
so a fresh download may not match the SHA-256 of the book's edition; the script says
so, and counts may then differ slightly from the printed ones.
"""
import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# file name the code expects: (Project Gutenberg eBook number, SHA-256 of the edition used)
BOOKS = {
    "alice_gutenberg_raw.txt": (11, "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e"),
    "looking_glass_gutenberg_raw.txt": (12, "e40a38f5da841f9fb6acc557355c447b7c2ccb56a208caf5fcf2b05081f14d42"),
    "hunting_of_the_snark_gutenberg_raw.txt": (13, "30d6d0a4fa246393ba85df5fa4fb3d0358bdf3f4784a8d0ebb96547275bc4f74"),
    "sylvie_and_bruno_gutenberg_raw.txt": (620, "d12fac177e54fb291073f24efa762478dfc0d4bc8de677e625eaf3f1d379bc5f"),
    "phantasmagoria_gutenberg_raw.txt": (651, "ad5cc3ca234f6fa1beac17f6e215baf21a4f36ac58d6b4bfe0c22e1c3a82ebe5"),
    "tangled_tale_gutenberg_raw.txt": (29042, "0ad1e38c495f1469368563bd75c3a5c24c36e3e5d5d75496d92a49b073af4dd8"),
    "sylvie_and_bruno_concluded_gutenberg_raw.txt": (48795, "5418ab237e518dba829ac0d571af2fdda0ad189a088b8cbb02ad4327dc9f0d63"),
}
URL = "https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def main():
    changed = 0
    for name, (n, want) in BOOKS.items():
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            print(f"downloading eBook #{n} -> 04-code/data/{name}")
            req = urllib.request.Request(URL.format(n=n), headers={"User-Agent": "tffp-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
                f.write(r.read())
        got = sha256(path)
        if got == want:
            print(f"ok       {name}")
        else:
            changed += 1
            print(f"CHANGED  {name}: SHA-256 {got[:12]}... differs from the book's edition {want[:12]}...")
    if changed:
        print(f"\n{changed} file(s) differ from the edition used in the book; counts may not match exactly.")
        sys.exit(1)
    print("\nAll texts match the editions used in the book.")


if __name__ == "__main__":
    main()
