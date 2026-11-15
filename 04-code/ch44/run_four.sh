#!/bin/bash
# Chapter 44 second batch: four-digit addition, direct against scratchpad, four threads per run; then the budget and vote
# tests on the two-digit and four-digit scratchpad models and the direct models.
cd "$(dirname "$0")"
NAME=direct4 FORMAT=direct DIGITS=4 THREADS=3 STEPS=1500 python3 add.py > outputs/direct4.txt 2>&1 &
NAME=pad4    FORMAT=pad    DIGITS=4 THREADS=3 STEPS=1500 python3 add.py > outputs/pad4.txt 2>&1 &
NAME=pad_s1865 THREADS=1 python3 budget.py > outputs/budget_pad_s1865.txt 2>&1 &
NAME=direct_s1865 THREADS=1 python3 budget.py > outputs/budget_direct_s1865.txt 2>&1 &
wait
NAME=pad4 THREADS=4 python3 budget.py > outputs/budget_pad4.txt 2>&1 &
NAME=direct4 THREADS=4 python3 budget.py > outputs/budget_direct4.txt 2>&1 &
wait
# Started by hand alongside the budget tests: the direct four-digit model trained three times as long.
# NAME=direct4_long FORMAT=direct DIGITS=4 THREADS=4 STEPS=4500 python3 add.py > outputs/direct4_long.txt
