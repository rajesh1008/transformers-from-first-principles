#!/bin/sh
# Added after order 7 reversed the sweep: the rest of the sweep at order 7, and three rates at order 11, all at once.
cd "$(dirname "$0")"
n=0; while read -r line; do n=$((n+1)); sh -c "$line" > outputs/log_orders_$n.txt 2>&1 & done < jobs_orders.txt; wait
