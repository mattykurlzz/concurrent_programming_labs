#!/bin/bash

plot_name="chunk_size_tests"

bins_dir="build"
measurments_dir="chunk_size"
tmp_file_name="tmp.log"
csv_name="$measurments_dir.csv"

num_threads=11
array_len_max=30000

if [[ ! -d "$measurments_dir" ]]; then 
    mkdir "$measurments_dir"
fi

if [[ -f "$measurments_dir/$csv_name" ]]; then 
    rm "$measurments_dir/$csv_name"
fi

touch "$measurments_dir/$csv_name"
echo "elements_count,time_taken,cycle_num,chunk_size" >> "$measurments_dir/$csv_name"

for array_len in 5000 15000 30000; do
    for bin_name in $(ls -1 $bins_dir); do
        if [[ $bin_name =~ .*_([0-9]+)_([0-9]+) ]]; then
            cycle_num="${BASH_REMATCH[1]}"
            chunk_size="${BASH_REMATCH[2]}"
        fi
        tmp_file="$measurments_dir/$tmp_file_name"
        ./build/$bin_name $array_len 1 $num_threads > "$tmp_file"

        line=$(grep "^N=[0-9]*\." "$tmp_file")
        value="${line#*: }"
        echo "$array_len,$value,$cycle_num,$chunk_size" >> "$measurments_dir/$csv_name"
    done
done    