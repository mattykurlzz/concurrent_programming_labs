#!/bin/bash

PLOT_NAME="speed_results"

PLOT_DIR="results/lab5_3_3_task_time"
TMP_FILE_NAME="tmp.log"
CSV_NAME="$PLOT_NAME.csv"
CHUNKS_NUM=20

THREADS=12
ARR_LEN_MAX=30000

if [[ ! -d "$PLOT_DIR" ]]; then 
    mkdir -p "$PLOT_DIR"
fi

# Create or clear the CSV file with new headers
echo "elements_count,time_taken,thread_count,type,generate_time,map_time,merge_time,sort_time,reduce_time,overhead_time" > "$PLOT_DIR/$CSV_NAME"

# Compile both versions with timing enabled
echo "Compiling OpenMP version..."
gcc -O3 -Wall -fopenmp -o build/lab5_3_3_omp lab5_3_3_omp.c -lm

echo "Compiling pthreads version..."
gcc -O3 -Wall -pthread -o build/lab5_3_3_pth lab5_3_3_pth.c -lm

# Function to parse timing data from output
parse_timing_data() {
    local output_file="$1"
    local type="$2"
    
    # Extract total time
    local total_line=$(grep "Total milliseconds passed:" "$output_file" | head -1)
    local total_time=""
    if [[ -n "$total_line" ]]; then
        total_time=$(echo "$total_line" | awk '{print $NF}')
    else
        # Fallback to old format
        total_line=$(grep "^N=[0-9]*\." "$output_file" | head -1)
        if [[ -n "$total_line" ]]; then
            total_time=$(echo "$total_line" | sed 's/.*: //' | sed 's/ ms.*//')
        fi
    fi
    
    # Extract stage times from the TOTAL line
    local generate_time="0"
    local map_time="0"
    local merge_time="0"
    local sort_time="0"
    local reduce_time="0"
    local overhead_time="0"
    
    # Try to find the TOTAL line in the timing statistics
    local total_stats_line=$(grep "^TOTAL" "$output_file" | head -1)
    if [[ -n "$total_stats_line" ]]; then
        # Parse the line with awk - split by | and get each field
        generate_time=$(echo "$total_stats_line" | awk -F'|' '{gsub(/ /, "", $3); print $3}')
        map_time=$(echo "$total_stats_line" | awk -F'|' '{gsub(/ /, "", $4); print $4}')
        merge_time=$(echo "$total_stats_line" | awk -F'|' '{gsub(/ /, "", $5); print $5}')
        sort_time=$(echo "$total_stats_line" | awk -F'|' '{gsub(/ /, "", $6); print $6}')
        reduce_time=$(echo "$total_stats_line" | awk -F'|' '{gsub(/ /, "", $7); print $7}')
        overhead_time=$(echo "$total_stats_line" | awk -F'|' '{gsub(/ /, "", $8); print $8}')
    else
        # Try alternative format with "TOTAL" at the beginning
        total_stats_line=$(grep "^[[:space:]]*TOTAL" "$output_file" | head -1)
        if [[ -n "$total_stats_line" ]]; then
            # Remove leading/trailing spaces and split
            clean_line=$(echo "$total_stats_line" | sed 's/^[[:space:]]*TOTAL[[:space:]]*|[[:space:]]*//')
            # Split by | and get values
            IFS='|' read -r -a times <<< "$clean_line"
            if [[ ${#times[@]} -ge 7 ]]; then
                generate_time=$(echo "${times[0]}" | tr -d ' ')
                map_time=$(echo "${times[1]}" | tr -d ' ')
                merge_time=$(echo "${times[2]}" | tr -d ' ')
                sort_time=$(echo "${times[3]}" | tr -d ' ')
                reduce_time=$(echo "${times[4]}" | tr -d ' ')
                overhead_time=$(echo "${times[5]}" | tr -d ' ')
            fi
        fi
    fi
    
    # Convert times from ms to seconds (for consistency)
    total_time=$(echo "$total_time / 1000" | bc -l)
    generate_time=$(echo "$generate_time / 1000" | bc -l)
    map_time=$(echo "$map_time / 1000" | bc -l)
    merge_time=$(echo "$merge_time / 1000" | bc -l)
    sort_time=$(echo "$sort_time / 1000" | bc -l)
    reduce_time=$(echo "$reduce_time / 1000" | bc -l)
    overhead_time=$(echo "$overhead_time / 1000" | bc -l)
    
    # Format to 6 decimal places
    total_time=$(printf "%.6f" "$total_time")
    generate_time=$(printf "%.6f" "$generate_time")
    map_time=$(printf "%.6f" "$map_time")
    merge_time=$(printf "%.6f" "$merge_time")
    sort_time=$(printf "%.6f" "$sort_time")
    reduce_time=$(printf "%.6f" "$reduce_time")
    overhead_time=$(printf "%.6f" "$overhead_time")
    
    echo "$total_time:$generate_time:$map_time:$merge_time:$sort_time:$reduce_time:$overhead_time"
}

echo "Starting benchmark runs..."

# Run benchmarks for different array sizes
for array_len in $(seq 1000 1000 15000); do
    echo "Testing array length: $array_len"
    
    for threadc in $(seq 1 1 12); do
        echo "  Threads: $threadc"
        
        # Run pthreads version
        tmp_file="$PLOT_DIR/$TMP_FILE_NAME.pth"
        ./build/lab5_3_3_pth "$array_len" 1 "$threadc" 0 > "$tmp_file" 2>&1
        
        timing_data=$(parse_timing_data "$tmp_file" "pythreads")
        total_time=$(echo "$timing_data" | cut -d':' -f1)
        generate_time=$(echo "$timing_data" | cut -d':' -f2)
        map_time=$(echo "$timing_data" | cut -d':' -f3)
        merge_time=$(echo "$timing_data" | cut -d':' -f4)
        sort_time=$(echo "$timing_data" | cut -d':' -f5)
        reduce_time=$(echo "$timing_data" | cut -d':' -f6)
        overhead_time=$(echo "$timing_data" | cut -d':' -f7)
        
        echo "$array_len,$total_time,$threadc,pythreads,$generate_time,$map_time,$merge_time,$sort_time,$reduce_time,$overhead_time" >> "$PLOT_DIR/$CSV_NAME"
        
        # Run OpenMP version
        tmp_file="$PLOT_DIR/$TMP_FILE_NAME.omp"
        ./build/lab5_3_3_omp "$array_len" 1 "$threadc" 0 > "$tmp_file" 2>&1
        
        timing_data=$(parse_timing_data "$tmp_file" "openmp")
        total_time=$(echo "$timing_data" | cut -d':' -f1)
        generate_time=$(echo "$timing_data" | cut -d':' -f2)
        map_time=$(echo "$timing_data" | cut -d':' -f3)
        merge_time=$(echo "$timing_data" | cut -d':' -f4)
        sort_time=$(echo "$timing_data" | cut -d':' -f5)
        reduce_time=$(echo "$timing_data" | cut -d':' -f6)
        overhead_time=$(echo "$timing_data" | cut -d':' -f7)
        
        echo "$array_len,$total_time,$threadc,openmp,$generate_time,$map_time,$merge_time,$sort_time,$reduce_time,$overhead_time" >> "$PLOT_DIR/$CSV_NAME"
        
        # Show progress
        echo "    pythreads: ${total_time}s, openmp: ${total_time}s"
    done
done

# Clean up temporary files
rm -f "$PLOT_DIR/$TMP_FILE_NAME"*
rm -f "$PLOT_DIR/$TMP_FILE_NAME.pth"
rm -f "$PLOT_DIR/$TMP_FILE_NAME.omp"

echo "Benchmark completed! Results saved to $PLOT_DIR/$CSV_NAME"

# Show summary
echo ""
echo "=== Summary of Results ==="
echo "Total runs: $(wc -l < "$PLOT_DIR/$CSV_NAME") lines in CSV"
echo "First few lines:"
head -5 "$PLOT_DIR/$CSV_NAME"
echo "..."
echo "Last few lines:"
tail -5 "$PLOT_DIR/$CSV_NAME"

# Create a summary statistics file
echo ""
echo "Creating summary statistics..."
cat > "$PLOT_DIR/summary_stats.txt" << EOF
Benchmark Summary
=================
Date: $(date)
Array sizes tested: 1000 to 15000 in steps of 1000
Thread counts tested: 1 to 12
Total data points: $(($(wc -l < "$PLOT_DIR/$CSV_NAME") - 1))

CSV Columns:
1. elements_count - Number of elements in array
2. time_taken - Total execution time (seconds)
3. thread_count - Number of threads used
4. type - Implementation type (pythreads or openmp)
5. generate_time - Time spent in generate phase (seconds)
6. map_time - Time spent in map phase (seconds)
7. merge_time - Time spent in merge phase (seconds)
8. sort_time - Time spent in sort phase (seconds)
9. reduce_time - Time spent in reduce phase (seconds)
10. overhead_time - Other overhead time (seconds)
EOF

echo "Summary saved to $PLOT_DIR/summary_stats.txt"