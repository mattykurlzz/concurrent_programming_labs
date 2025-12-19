#!/bin/bash

plot_name="serial_tests_name"

bins_dir="build"
measurments_dir="serial"
tmp_file_name="tmp.log"
csv_name="$measurments_dir.csv"
# params_name="$(plot_name)_params"

max_threads=12
array_len_max=30000

if [[ ! -d "$measurments_dir" ]]; then 
    mkdir "$measurments_dir"
fi

if [[ -f "$measurments_dir/$csv_name" ]]; then 
    rm "$measurments_dir/$csv_name"
fi

touch "$measurments_dir/$csv_name"
echo "thread_count,elements_count,time_taken,letter" >> "$measurments_dir/$csv_name"                

for array_len in 5000 15000 30000; do
    for bin_name in $(ls -1 $bins_dir); do
        letter=$(echo "$bin_name" | grep -o '[A-Z]')
        for threadc in $(seq 2 $max_threads); do
            tmp_file="$measurments_dir/$tmp_file_name"
            ./build/$bin_name $array_len 1 $threadc > "$tmp_file"
            
            line=$(grep "^N=[0-9]*\." "$tmp_file")
            value="${line#*: }"
            echo "$threadc,$array_len,$value,$letter" >> "$measurments_dir/$csv_name"                
        done
    done
done    