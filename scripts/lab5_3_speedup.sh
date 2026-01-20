#!/bin/bash

PLOT_NAME="speed_results"

PLOT_DIR="results/lab5_3_task_time"
TMP_FILE_NAME="tmp.log"
CSV_NAME="$PLOT_NAME.csv"
CHUNKS_NUM=20

THREADS=12
ARR_LEN_MAX=30000

if [[ ! -d "$PLOT_DIR" ]]; then 
    mkdir "$PLOT_DIR"
fi

if [[ -f "$PLOT_DIR/$CSV_NAME" ]]; then 
    rm "$PLOT_DIR/$CSV_NAME"
fi

touch "$PLOT_DIR/$CSV_NAME"
echo "elements_count,time_taken,thread_count,type" >> "$PLOT_DIR/$CSV_NAME"

gcc -O3 -Wall -fopenmp -o build/lab5_3_3_omp lab5_3_3_omp.c -lm
gcc -O3 -Wall -pthread -o build/lab5_3_3_pth lab5_3_3_pth.c -lm

for array_len in $(seq 1000 1000 15000); do
    for threadc in $(seq 1 1 12); do
        tmp_file="$PLOT_DIR/$TMP_FILE_NAME"
        ./build/lab5_3_3_pth $array_len 1 $threadc > "$tmp_file"

        line=$(grep "^N=[0-9]*\." "$tmp_file")
        value="${line#*: }"
        echo "$array_len,$value,$threadc,pythreads" >> "$PLOT_DIR/$CSV_NAME"

        tmp_file="$PLOT_DIR/$TMP_FILE_NAME"
        ./build/lab5_3_3_omp $array_len 1 $threadc > "$tmp_file"

        line=$(grep "^N=[0-9]*\." "$tmp_file")
        value="${line#*: }"
        echo "$array_len,$value,$threadc,openmp" >> "$PLOT_DIR/$CSV_NAME"
    done
done    