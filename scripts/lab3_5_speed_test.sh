#!/bin/bash

PLOT_NAME="speed_results"

PLOT_DIR="plot"
TMP_FILE_NAME="tmp.log"
CSV_NAME="$PLOT_NAME.csv"

THREADS=12
ARR_LEN_MAX=30000

if [[ ! -d "$PLOT_DIR" ]]; then 
    mkdir "$PLOT_DIR"
fi

if [[ -f "$PLOT_DIR/$CSV_NAME" ]]; then 
    rm "$PLOT_DIR/$CSV_NAME"
fi

touch "$PLOT_DIR/$CSV_NAME"
echo "thread_count,elements_count,time_taken" >> "$PLOT_DIR/$CSV_NAME"                

for array_len in 5000 15000 30000; do
    for threadc in $(seq 1 $THREADS); do
        tmp_file="$PLOT_DIR/$TMP_FILE_NAME"
        ./lab3_parallel_fopenmp $array_len 1 $threadc > "$tmp_file"
        
        line=$(grep "^N=[0-9]*\." "$tmp_file")
        value="${line#*: }"
        echo "$threadc,$array_len,$value" >> "$PLOT_DIR/$CSV_NAME"                
    done
done    