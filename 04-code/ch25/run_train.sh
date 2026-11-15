#!/bin/sh
# Chapter 25's five new training runs (about 16 minutes, five at a time, one thread each): the 1e-3 WSD trunk of
# Chapter 24 with branches at 1,000 and 2,000 steps (finished at 1,250 and 2,500), with AdamW decay 0.1 or none,
# under reading orders 1865, 7 and 11.  Chapter 24's run (no decay, order 1865) is the sixth; its tables are read from
# ../ch24/outputs/ckpt/.  WD=0 is written as no decay (train_tiny.py treats 0 as plain Adam).
cd "$(dirname "$0")"
while read -r a b; do
  tag=$(echo "$a $b" | tr " =" "__")
  env $a $b python3 train_tiny.py wsd 1e-3 2000 1000,2000 > outputs/log_train_$tag.txt 2>&1 &
done < jobs_train.txt
wait
