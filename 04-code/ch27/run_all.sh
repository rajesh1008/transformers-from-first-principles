#!/bin/sh
# Chapter 27: every script in order (about 10 minutes, one thread).  Needs ../ch24/outputs/ckpt/wsd_1e-3_2000_b2000.pt.
set -e
cd "$(dirname "$0")"
python3 -u generate_basics.py > outputs/generate_basics.txt
python3 -u mat_rank.py        > outputs/mat_rank.txt
python3 -u rules.py           > outputs/rules.txt
python3 -u experiments.py     > outputs/experiments.txt
python3 -u stopping.py        > outputs/stopping.txt
python3 -u heldout_words.py   > outputs/heldout_words.txt
python3 -u lab27.py           > outputs/lab27.txt
python3 ../../03-figures/src/ch27_figs.py
