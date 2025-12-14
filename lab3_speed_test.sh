#!/bin/bash

plot_name="speed_results"

plot_dir="plot"
tmp_file_name="tmp.log"
csv_name="$plot_name.csv"
# params_name="$(plot_name)_params"

max_threads=12
array_len_max=50000

if [[ ! -d "$plot_dir" ]]; then 
    mkdir "$plot_dir"
fi

if [[ -f "$plot_dir/$csv_name" ]]; then 
    rm "$plot_dir/$csv_name"
fi
# if [ -f "$plot_dir/$params_name"]; then 
#     rm "$plot_dir/$params_name"
# fi

touch "$plot_dir/$csv_name"
echo "thread_count,elements_count,time_taken" >> "$plot_dir/$csv_name"                

for array_len in $(seq 5000 5000 $array_len_max); do
    for threadc in $(seq 1 $max_threads); do
        tmp_file="$plot_dir/$tmp_file_name"
        ./lab3_parallel_fopenmp $array_len 1 $threadc > "$tmp_file"
        
        line=$(grep "^N=[0-9]*\." "$tmp_file")
        value="${line#*: }"
        echo "$threadc,$array_len,$value" >> "$plot_dir/$csv_name"                
    done
done    