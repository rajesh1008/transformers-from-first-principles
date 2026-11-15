#!/bin/sh
# Every run of Chapter 24, eight at a time, one thread each (jobs.txt).  Logs to outputs/log_*.txt.
cd "$(dirname "$0")"
xargs -P 8 -L 1 sh -c 'tag=$(echo "$@" | tr " ," "__" | sed "s/.py//"); python3 "$@" > outputs/log_$tag.txt 2>&1' _ < jobs.txt
