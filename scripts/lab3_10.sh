#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <program> [args...]"
    exit 1
fi

measurments_dir="cpu_load"

if [[ ! -d "$measurments_dir" ]]; then 
    mkdir "$measurments_dir"
fi

PROGRAM=$1
shift
ARGS="$@"
CSV_FILE="monitor.csv"
PID_FILE="/$measurments_dir/tmp.txt"

echo "timestamp_ms,cpu_percent,memory_percent,threads" > "$measurments_dir/$CSV_FILE"

$PROGRAM $ARGS &
PROG_PID=$!
echo "PID: $PROG_PID"

cleanup() {
    kill $PROG_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

START_MS=$(date +%s%3N)
while kill -0 $PROG_PID 2>/dev/null; do
    CURRENT_MS=$(date +%s%3N)
    ELAPSED=$((CURRENT_MS - START_MS))
    
    STATS=$(ps -p $PROG_PID -o %cpu,%mem,thcount --no-headers 2>/dev/null)
    if [ -n "$STATS" ]; then
        CPU=$(echo $STATS | awk '{print $1}')
        MEM=$(echo $STATS | awk '{print $2}')
        THREADS=$(echo $STATS | awk '{print $3}')
        echo "$ELAPSED,$CPU,$MEM,$THREADS" >> "$measurments_dir/$CSV_FILE"
    fi
    
    sleep 0.001
done