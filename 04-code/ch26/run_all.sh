#!/bin/sh
# Chapter 26, everything after the training runs (run ./run_seeds.sh first: about 12 minutes for 26 runs of 400 steps
# five at a time, and 12 minutes for Chapter 24's run repeated, on its own core).  One thread unless a script says otherwise.
cd "$(dirname "$0")"
python3 seeds_demo.py    > outputs/seeds_demo.txt
python3 fp_order.py      > outputs/fp_order.txt
python3 analyse.py       > outputs/analyse.txt
python3 hosted_summary.py > outputs/hosted_summary.txt     # reads outputs/repeat_hosted.json; repeat_hosted.py makes the API calls
python3 lab26.py         > outputs/lab26.txt
python3 solutions.py     > outputs/solutions.txt
(cd ../.. && python3 03-figures/src/ch26_figs.py)
