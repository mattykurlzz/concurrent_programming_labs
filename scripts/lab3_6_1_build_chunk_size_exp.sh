#!/bin/bash

CYCLE_INDEX_NUM=6
CHUNKS_NUM=24

rm -rf ./build/*

for ((CYCLE_INDEX=1; CYCLE_INDEX<=6; CYCLE_INDEX++)); do
    for ((CHUNK=1; CHUNK<=CHUNKS_NUM; CHUNK++)); do
        echo "Building chunk $CHUNK for cycle $CYCLE_INDEX"
        gcc -O3 -Wall -fopenmp -DCHUNK_SIZE=$((CHUNK+1)) -DCYCLE_INDEX=${CYCLE_INDEX} -o build/lab3_${CYCLE_INDEX}_${CHUNK} lab3_6_1.c -lm
    done
done
