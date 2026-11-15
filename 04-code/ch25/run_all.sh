#!/bin/sh
# Chapter 25 end to end, one thread, 32-bit.  run_train.sh first (five runs, about 12 minutes each, five at a time),
# then the measurements in order.  judge.py calls a hosted model and needs ANTHROPIC_API_KEY in ../../.env; it is
# skipped if that file is missing.  Nothing here prints or writes the key.
set -e
cd "$(dirname "$0")"
mkdir -p outputs
./run_train.sh
python3 evaluate.py    > outputs/evaluate.txt
python3 contaminate.py > outputs/contaminate.txt
python3 bench.py       > outputs/bench.txt
python3 compare.py     > outputs/compare.txt
if [ -f ../../.env ]; then python3 judge.py > outputs/judge.txt; python3 judge_summary.py > /dev/null; fi
python3 lab25.py       > outputs/lab25.txt
python3 solutions.py   > outputs/solutions.txt
python3 ../../03-figures/src/ch25_figs.py
