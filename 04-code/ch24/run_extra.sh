#!/bin/sh
# The five runs added after the sweep (jobs_extra.txt: AdamW, cosine floor, second reading order), three at a time.
cd "$(dirname "$0")"
n=0; while read -r line; do n=$((n+1)); sh -c "$line" > outputs/log_extra_$n.txt 2>&1 & if [ $((n % 3)) -eq 0 ]; then wait; fi; done < jobs_extra.txt; wait
