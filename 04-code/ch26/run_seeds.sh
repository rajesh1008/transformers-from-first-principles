#!/bin/sh
# Chapter 26's fifteen 400-step runs (jobs_seeds.txt), five at a time, plus Chapter 24's 2,500-step run repeated
# from nothing (train_tiny.py wsd 1e-3 2000 2000; about 15 minutes on its own core).  On 11 Nov 2026 the repeat was
# started first and the grid relaunched three minutes later (the first grid launch did not save its tables).
cd "$(dirname "$0")"
[ "$1" = "grid" ] || python3 train_tiny.py wsd 1e-3 2000 2000 > outputs/log_repeat_ch24.txt 2>&1 &
cat jobs_seeds.txt | xargs -P 5 -I{} sh -c 'env {} python3 train_seeds.py >> outputs/log_seeds.txt 2>&1'
wait
# second grid (added at 10:03 IST once the first was seen to take two minutes a wave): two more seeds, a second learning
# rate on five seeds, and init-only / order-only seeds 21 and 42.
cat jobs_seeds2.txt | xargs -P 5 -I{} sh -c 'env {} python3 train_seeds.py >> outputs/log_seeds.txt 2>&1'
