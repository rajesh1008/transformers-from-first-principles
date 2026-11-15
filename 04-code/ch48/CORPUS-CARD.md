# Corpus card: Chapter 48 capstone

Recorded 12 November 2026 for the run in `run_main.sh` (seed 1865). Re-created by `prep48.py`; sizes from `outputs/prep48.txt`.

**Sources.** Fourteen books from Project Gutenberg (https://www.gutenberg.org/), downloaded by `prep48.py` in this order (Gutenberg ids in `corpus/books.json`): Pride and Prejudice, Frankenstein, The Adventures of Sherlock Holmes, A Tale of Two Cities, Great Expectations, Wuthering Heights, Jane Eyre, Dracula, The Picture of Dorian Gray, Emma, Sense and Sensibility, Persuasion, Mansfield Park, Oliver Twist.

**Terms.** Every book is in the public domain in the United States, where Project Gutenberg operates. Every author died before 1956 (Austen 1817, Shelley 1851, C. Brontë 1855, E. Brontë 1848, Dickens 1870, Wilde 1900, Stoker 1912, Conan Doyle 1930; held-out Wells 1946, Carroll 1898), so the works are also out of copyright where protection is the author's life plus seventy years. Project Gutenberg's header, footer and licence text are removed, because the Project Gutenberg name is not used with the derived text.

**Size.** 10,381,201 bytes of training text (10,198,749 characters, 2,719,058 tokens with the 8,192-token BPE of `tok48.py`).

**Removed.** Gutenberg header, footer and licence; nothing else (no deduplication, no filtering, no normalisation of quotation marks or dashes).

**Held out.** *The Time Machine* (H. G. Wells, 181,475 bytes), never in training; *Through the Looking-Glass* (Lewis Carroll, 164,250 characters), the Chapter 22/25 text on which Tiny was measured, for a comparison in bits per character.

**Lean.** Five of the fourteen books are Jane Austen's and three Charles Dickens's: almost half the text is early-nineteenth-century English by two writers, with dialogue, titles (Mr., Miss) and hard line breaks at about seventy characters. No Carroll and no Wells in training. No text after 1900. Expect the model to write that.
