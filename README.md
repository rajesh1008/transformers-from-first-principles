# Transformers from First Principles: companion code

Code for **_Transformers from First Principles: Build Attention, Step by Step, with Only High-School Algebra_** by **Rajesh Ranjan Mahapatra** (Math & CS from First Principles, November 2026).

The book builds a small transformer, Tiny, from dot products and weighted averages, trains it on Lewis Carroll's *Alice's Adventures in Wonderland*, makes it write, and then uses it to explain what larger models do. Every number printed in the book came from the scripts in this repository, and each chapter folder keeps the outputs those scripts printed, so you can check your run against ours line by line.

## What is inside

| Path | What it holds |
|---|---|
| `04-code/ch01` to `04-code/ch50` | One folder per chapter: the scripts the chapter quotes, its lab, `solutions.py` for the numerical exercises where there is one, `RUN.txt` (what to run and in what order) and `outputs/` (what the book's runs printed) |
| `04-code/tiny/` | The `tiny` package, built up chapter by chapter: tokeniser, embeddings, attention, feed-forward block, model, training loop, generation, cache, quantisation and serving |
| `04-code/data/` | The seven public-domain Project Gutenberg texts (Lewis Carroll), in the exact editions the book used, and `fetch_data.py`, which checks them |
| `03-figures/src/` | The scripts that draw every figure in the book (`chNN_figs.py`, with the shared style in `style.py`) |
| `03-figures/cover/src/` | The script that draws the cover |

## Chapter-to-folder map

**Part I: Words Become Numbers**

| Chapter | Title | Folder |
|---|---|---|
| 1 | What a Language Model Is | [`04-code/ch01`](04-code/ch01) |
| 2 | Tokens | [`04-code/ch02`](04-code/ch02) |
| 3 | Byte-Pair Encoding in Depth | [`04-code/ch03`](04-code/ch03) |
| 4 | Embeddings | [`04-code/ch04`](04-code/ch04) |
| 5 | Position: Learned and Sinusoidal | [`04-code/ch05`](04-code/ch05) |
| 6 | Rotary Positions and Relative Order | [`04-code/ch06`](04-code/ch06) |

**Part II: Attention, by Hand**

| Chapter | Title | Folder |
|---|---|---|
| 7 | Many at Once | [`04-code/ch07`](04-code/ch07) |
| 8 | One Head of Attention | [`04-code/ch08`](04-code/ch08) |
| 9 | Many Heads | [`04-code/ch09`](04-code/ch09) |
| 10 | Attention Variants: Grouped-Query and Sliding-Window | [`04-code/ch10`](04-code/ch10) |
| 11 | Attention Efficiency: Memory Traffic, Tiling and Why the Hardware Shape Matters | [`04-code/ch11`](04-code/ch11) |
| 12 | The Feed-Forward Block as Memory | [`04-code/ch12`](04-code/ch12) |
| 13 | Normalisation and Residuals | [`04-code/ch13`](04-code/ch13) |
| 14 | Stacking Layers | [`04-code/ch14`](04-code/ch14) |
| 15 | Workshop: Tracing the Sentence Through Tiny by Hand | [`04-code/ch15`](04-code/ch15) |
| 16 | Reading a Config File: From Hyperparameters to Parameter Count, by Hand | [`04-code/ch16`](04-code/ch16) |

**Part III: Learning**

| Chapter | Title | Folder |
|---|---|---|
| 17 | Learning as Nudging: Gradient Descent from One Parameter Up | [`04-code/ch17`](04-code/ch17) |
| 18 | Backpropagation by Hand Through One Head of Attention | [`04-code/ch18`](04-code/ch18) |
| 19 | Initialisation and Why Training Is Stable | [`04-code/ch19`](04-code/ch19) |
| 20 | Optimisers and Learning-Rate Schedules | [`04-code/ch20`](04-code/ch20) |
| 21 | Mixed Precision, Memory and Where the Training Budget Goes | [`04-code/ch21`](04-code/ch21) |
| 22 | Data Pipelines and the Curriculum | [`04-code/ch22`](04-code/ch22) |
| 23 | Scaling Laws in Miniature: Train Five Tinys and Fit the Curve | [`04-code/ch23`](04-code/ch23) |
| 24 | Training Tiny | [`04-code/ch24`](04-code/ch24) |
| 25 | Evaluating a Language Model | [`04-code/ch25`](04-code/ch25) |
| 26 | Reproducibility: Seeds, Nondeterminism and Why Two Runs Differ | [`04-code/ch26`](04-code/ch26) |

**Part IV: Speaking**

| Chapter | Title | Folder |
|---|---|---|
| 27 | Generating Text | [`04-code/ch27`](04-code/ch27) |
| 28 | Decoding Strategies in Depth | [`04-code/ch28`](04-code/ch28) |
| 29 | Loading the Real Thing | [`04-code/ch29`](04-code/ch29) |
| 30 | The KV Cache, Cost and Speed | [`04-code/ch30`](04-code/ch30) |
| 31 | Serving: Batching, Paged Memory and the Real Cost of a Token | [`04-code/ch31`](04-code/ch31) |
| 32 | Speculative Decoding | [`04-code/ch32`](04-code/ch32) |
| 33 | Quantisation and Running Tiny Small | [`04-code/ch33`](04-code/ch33) |
| 34 | Distillation and Pruning: Making Tiny Smaller Without Starting Over | [`04-code/ch34`](04-code/ch34) |

**Part V: What the Build Explains**

| Chapter | Title | Folder |
|---|---|---|
| 35 | Context Windows and Their Limits | [`04-code/ch35`](04-code/ch35) |
| 36 | In-Context Learning and Retrieval, Mechanically | [`04-code/ch36`](04-code/ch36) |
| 37 | Why It Makes Things Up | [`04-code/ch37`](04-code/ch37) |
| 38 | From Predictor to Assistant | [`04-code/ch38`](04-code/ch38) |
| 39 | Fine-Tuning in Miniature: Full, Low-Rank and Adapters on Tiny | [`04-code/ch39`](04-code/ch39) |
| 40 | Interpretability: Probes and Attention Maps | [`04-code/ch40`](04-code/ch40) |
| 41 | Circuits in Tiny: Induction Heads, the Logit Lens and Sparse Features | [`04-code/ch41`](04-code/ch41) |

**Part VI: Beyond the Vanilla Transformer**

| Chapter | Title | Folder |
|---|---|---|
| 42 | Mixture of Experts in Miniature | [`04-code/ch42`](04-code/ch42) |
| 43 | The Same Machine Looks at Pictures | [`04-code/ch43`](04-code/ch43) |
| 44 | Reasoning Models, Test-Time Compute and State-Space Models | [`04-code/ch44`](04-code/ch44) |
| 45 | Distributed Training on Paper: Splitting a Model Across Machines Without a Cluster | [`04-code/ch45`](04-code/ch45) |
| 46 | Common Misconceptions, Corrected | [`04-code/ch46`](04-code/ch46) |
| 47 | The Machine-Learning Engineer's Interview, From Both Sides | [`04-code/ch47`](04-code/ch47) |

**Part VII: The Capstone and What Comes Next**

| Chapter | Title | Folder |
|---|---|---|
| 48 | Capstone: Train a 10M-Parameter Model on a Laptop | [`04-code/ch48`](04-code/ch48) |
| 49 | Deploying Tiny: Export, an API, a Dashboard and the First Week of Traffic | [`04-code/ch49`](04-code/ch49) |
| 50 | Where This Is Going | [`04-code/ch50`](04-code/ch50) |

## Setting up

The book's runs used **Python 3.13.5** on an ordinary CPU with no graphics card; Python 3.12 or later should work. **Nothing needs a GPU**: every script runs on the CPU. Most finish in seconds; a few training runs, such as Chapter 48's capstone, take hours, and `RUN.txt` gives the time each took on the book's machine.

```bash
git clone https://github.com/rajesh1008/transformers-from-first-principles.git
cd transformers-from-first-principles
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python3 04-code/data/fetch_data.py      # optional: checks the Gutenberg texts are the book's editions
```

`requirements.txt` pins the versions used for the book. Chapters 1 to 16 need only NumPy and matplotlib (plus tiktoken and regex in Chapters 4, 5 and 9); PyTorch arrives in Chapter 17.

The training texts are stored in the repository in the exact editions the book used, because Project Gutenberg corrects its files from time to time and a fresh download can shift counts such as Chapter 1's 26,776 words. `fetch_data.py` checks each file's SHA-256 and re-downloads any that are missing.

Published models (GPT-2 small, SmolLM2-135M, Qwen3-0.6B and others) are downloaded the first time a chapter needs them; the chapter's `RUN.txt` says where they go. GPT-2 small's weight file alone is 548 MB.

## Running a chapter

Every script is run **from its own chapter folder**, exactly as the book's *Run* lines say:

```bash
cd 04-code/ch01
python3 bigram.py                     # Chapter 1's checkpoint
python3 lab_smoothing.py
```

Each folder's `RUN.txt` lists its scripts in order, with any arguments, the output file each one writes and, for longer runs, how long it took on the book's machine. Compare what you get with the matching file in `outputs/`.

**Seeds.** Every seeded script uses the seed **1865**, the year *Alice* was published, and the shared set-up in most chapters pins NumPy and PyTorch to one CPU thread in 32-bit arithmetic. On the same versions you should see the same numbers. Timings belong to the book's machine and will differ on yours, and mixed-precision (BF16) runs in Chapters 21 to 23 do not always repeat to the last digit.

**Figures.** From the repository's top folder, `python3 03-figures/src/ch01_figs.py` redraws Chapter 1's figures into `03-figures/ch01/` (the same pattern works for every chapter). The book uses the Inter typeface; without it matplotlib falls back to DejaVu Sans.

**Re-created artefacts.** Checkpoints, token shards, the capstone's 21 MB corpus and model weights are not stored here, because the scripts that need them re-create them. `RUN.txt` says which step does it.

## Chapters that call a hosted model

Some chapters send prompts to Anthropic's API (Claude Haiku 4.5 and Claude Haiku 5.5), and a few labs check for the key file even when they make no call. Those scripts read your key from a file called `.env` in the **repository's top folder**:

```bash
cp .env.example .env
# then edit .env so that it reads:  ANTHROPIC_API_KEY=<your key>
```

If you keep the key in an environment variable instead, write it into the file with `echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" > .env`. The file is listed in `.gitignore`; **never commit it**. The scripts read the key into memory only and never print it. The chapters involved are 25 to 40, 43, 44 and 47.

## Spelling and conventions

The prose in the book, and in the comments and docstrings here, is in **British English** (*quantisation*, *colour*, *behaviour*), except for code identifiers, quoted output, paper titles and the field's spelling *tokenizer*.

## Licence and citation

The code is released under the [MIT licence](LICENSE), © 2026 Rajesh Ranjan Mahapatra. The book's text is not part of this licence. The Project Gutenberg texts are in the public domain in the United States. To cite the book or the code, use [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button reads it).
