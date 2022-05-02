#!/usr/bin/env sh
echo "Running for five generations for profiling"
mkdir -p tmp
python -m cProfile -o tmp/runtime.prof\
    -m bitwise_challenge_2022_2 --max-gen 5\
    --metric-log tmp/log.txt --xpath tmp/X.npy
snakeviz tmp/runtime.prof
