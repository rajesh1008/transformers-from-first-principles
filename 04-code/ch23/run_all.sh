#!/bin/sh
# Every training run of Chapter 23, seven at a time, one thread each.  jobs.txt lists SIZE STEPS ORDER_SEED [LR].
cd "$(dirname "$0")"
xargs -P 7 -L 1 sh -c 'tag=$(echo "$@" | tr " " "_"); python3 train_scale.py "$@" > outputs/log_$tag.txt 2>&1' _ < jobs.txt
