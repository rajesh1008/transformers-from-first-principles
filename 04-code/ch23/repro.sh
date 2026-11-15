#!/bin/sh
# Two more runs each of s3 and s5 (384 steps, order 1865, lr 1e-3) with the configuration unchanged, to measure
# run-to-run differences of BF16 products on the CPU (repro_probe.py shows 32-bit products repeat exactly).
cd "$(dirname "$0")"
while pgrep -f run_all.sh > /dev/null; do sleep 5; done
for r in _r2 _r3; do
  TAG_SUFFIX=$r python3 train_scale.py s3 384 1865 > outputs/log_s3_384$r.txt 2>&1 &
  TAG_SUFFIX=$r python3 train_scale.py s5 384 1865 > outputs/log_s5_384$r.txt 2>&1 &
  TAG_SUFFIX=$r python3 train_scale.py s1 384 1865 > outputs/log_s1_384$r.txt 2>&1 &
done
wait
