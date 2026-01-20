#!/bin/bash

PLOT_NAME="speed_results"

PLOT_DIR="results/lab5_3_5_memory_usage"
TMP_FILE_NAME="tmp.log"
TIME_OUTPUT="time_output.txt"
CSV_NAME="$PLOT_NAME.csv"
CHUNKS_NUM=20

THREADS=12
array_len=20000

if [[ ! -d "$PLOT_DIR" ]]; then 
    mkdir -p "$PLOT_DIR"
fi

# Update CSV header to include memory usage
echo "elements_count,time_taken,thread_count,type,memory_kb" > "$PLOT_DIR/$CSV_NAME"

gcc -O3 -Wall -fopenmp -o build/lab5_3_omp lab5_3_omp.c -lm
gcc -O3 -Wall -pthread -o build/lab5_3_pth lab5_2_pth.c -lm

# Function to measure memory usage of a process
measure_memory_usage() {
    local pid=$1
    local max_memory=0
    
    # Monitor process memory usage while it's running
    while kill -0 "$pid" 2>/dev/null; do
        # Get resident set size (RSS) in KB
        local mem_usage=$(ps -o rss= -p "$pid" 2>/dev/null)
        if [[ -n "$mem_usage" ]] && [[ "$mem_usage" -gt "$max_memory" ]]; then
            max_memory=$mem_usage
        fi
        sleep 0.1  # Check every 100ms
    done
    
    echo "$max_memory"
}

# Function to run a benchmark and measure memory
run_benchmark() {
    local program="$1"
    local array_len="$2"
    local threadc="$3"
    local type="$4"
    
    local tmp_file="$PLOT_DIR/$TMP_FILE_NAME.$type.$array_len.$threadc"
    
    # Run the program in background
    "$program" "$array_len" 1 "$threadc" > "$tmp_file" 2>&1 &
    local pid=$!
    
    # Measure memory usage
    local memory_kb=$(measure_memory_usage $pid)
    
    # Wait for the process to complete
    wait $pid
    
    # Extract time from output
    local line=$(grep "^N=[0-9]*\." "$tmp_file")
    local value=""
    if [[ -n "$line" ]]; then
        value="${line#*: }"
        # Remove " ms" suffix if present
        value="${value/ ms/}"
    else
        # Try alternative format
        line=$(grep "Milliseconds passed:" "$tmp_file")
        if [[ -n "$line" ]]; then
            value=$(echo "$line" | grep -o '[0-9]*')
        fi
    fi
    
    # Convert time from ms to seconds if needed
    if [[ "$value" -gt 1000 ]]; then
        # Assume it's in ms if value > 1000
        value=$(echo "scale=6; $value / 1000" | bc)
    fi
    
    # Clean up temp file
    rm -f "$tmp_file"
    
    # Output results
    echo "$array_len,$value,$threadc,$type,$memory_kb"
}

echo "Starting benchmarks with memory measurement..."

echo "Testing array length: $array_len"

for threadc in $(seq 1 10 201); do
    echo "  Threads: $threadc"
    
    # Run pthreads version with memory measurement
    echo "    Running pthreads..."
    result=$(run_benchmark "./build/lab5_3_pth" "$array_len" "$threadc" "pythreads")
    echo "$result" >> "$PLOT_DIR/$CSV_NAME"
    
    # Run OpenMP version with memory measurement
    echo "    Running OpenMP..."
    result=$(run_benchmark "./build/lab5_3_omp" "$array_len" "$threadc" "openmp")
    echo "$result" >> "$PLOT_DIR/$CSV_NAME"
done

echo "Benchmark completed! Results saved to $PLOT_DIR/$CSV_NAME"

# Show summary
echo ""
echo "=== Summary ==="
echo "CSV format: elements_count,time_taken(seconds),thread_count,type,memory_kb(KB)"
echo "Total entries: $(($(wc -l < "$PLOT_DIR/$CSV_NAME") - 1))"
echo ""
echo "First few entries:"
head -5 "$PLOT_DIR/$CSV_NAME"