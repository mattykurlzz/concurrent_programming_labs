#!/bin/bash

PLOT_NAME="speed_results"

PLOT_DIR="results/thread_overhead"
TMP_FILE_NAME="tmp.log"
CSV_NAME="$PLOT_NAME.csv"
CHUNKS_NUM=20

THREADS=20
ARR_LEN_MAX=30000

if [[ ! -d "$PLOT_DIR" ]]; then 
    mkdir "$PLOT_DIR"
fi

if [[ -f "$PLOT_DIR/$CSV_NAME" ]]; then 
    rm "$PLOT_DIR/$CSV_NAME"
fi

touch "$PLOT_DIR/$CSV_NAME"
echo "elements_count,time_taken,cycle_index,schedule_type,thread_count" >> "$PLOT_DIR/$CSV_NAME"

for ((CYCLE_INDEX=1; CYCLE_INDEX<=6; CYCLE_INDEX++)); do
    for TYPE in static dynamic guided; do
        for threadc in 1 2 11; do
            gcc -O3 -Wall -fopenmp -DCHUNK_SIZE=1 -DSCHEDULE_TYPE=$TYPE -DCYCLE_INDEX=${CYCLE_INDEX} -o build/lab3_${CYCLE_INDEX}_${TYPE}_${threadc} lab3_13.c -lm
        done
    done
done


for array_len in $(seq 50 50 800); do
    for ((CYCLE_INDEX=1; CYCLE_INDEX<=6; CYCLE_INDEX++)); do
        for TYPE in static dynamic guided; do
            for threadc in 1 2 11; do
                tmp_file="$PLOT_DIR/$TMP_FILE_NAME"
                ./build/lab3_${CYCLE_INDEX}_${TYPE}_${threadc} $array_len 1 $threadc > "$tmp_file"

                line=$(grep "^N=[0-9]*\." "$tmp_file")
                value="${line#*: }"
                echo "$array_len,$value,$CYCLE_INDEX,$TYPE,$threadc" >> "$PLOT_DIR/$CSV_NAME"
            done
        done
    done
done    