#!/bin/bash
# Chapter 44: direct answers against a scratchpad, two seeds each, two threads per run.
cd "$(dirname "$0")"
for s in 1865 2; do
  NAME=direct_s$s FORMAT=direct SEED=$s THREADS=2 python3 add.py > outputs/direct_s$s.txt 2>&1 &
  NAME=pad_s$s    FORMAT=pad    SEED=$s THREADS=2 python3 add.py > outputs/pad_s$s.txt 2>&1 &
done
wait
