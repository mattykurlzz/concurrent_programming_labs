#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>

#define EULERS (2.718281828459045)
#define MINIMAL_DIVISOR 0.001

// Timing structure to store measurements
typedef struct {
    double total_time;
    double generate_time;
    double map_time;
    double merge_time;
    double sort_time;
    double reduce_time;
    double overhead_time;
    int iteration_count;
} TimingStats;

// Global timing statistics (one per iteration)
TimingStats *timing_stats = NULL;
int current_iteration = 0;
pthread_mutex_t timing_mutex;

typedef struct {
    uint32_t id;
    uint32_t start;
    uint32_t end;
    uint32_t len;
    uint32_t A;
    uint32_t iteration;
    int fixed;
    float *M1;
    float *M2;
    float *M2_copy;
    int iteration_idx;  // Add iteration index to thread data
} thread_data_t;

pthread_mutex_t progress_mutex;
pthread_mutex_t compare_mutex;
int progress = 0;
int finished = 0;
float global_compare = 0;

double get_wtime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

unsigned int hash_position(uint32_t position, uint32_t iteration) {
    unsigned int h = position;
    h ^= iteration << 16;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

void *generate_thread(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    for (uint32_t i = data->start; i < data->end; i++) {
        if (data->fixed) {
            data->M1[i] = data->M1[i % 2];
            if (i < data->len / 2) {
                data->M2[i] = data->M1[i % 2] * data->A;
            }
        } else {
            unsigned int local_seed = hash_position(i, data->iteration);
            data->M1[i] = (rand_r(&local_seed) % data->A) + 1;
            if (i < data->len / 2) {
                local_seed = hash_position(i + 0x10000, data->iteration);
                data->M2[i] = (rand_r(&local_seed) % (9 * data->A + 1)) + data->A;
            }
        }
    }
    return NULL;
}

void generate(float **M1_p, float **M2_p, const uint32_t len, const uint32_t A,
              unsigned int seed, const int fixed, int threads_num, int iteration_idx) {
    double start_time = get_wtime();
    
    *M1_p = (float *)malloc(sizeof(float) * len);
    *M2_p = (float *)malloc(sizeof(float) * (len / 2));

    if (*M1_p == NULL || *M2_p == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    if (fixed && (len > 1)) {
        (*M1_p)[0] = 1.2;
        (*M1_p)[1] = 3;
    }

    pthread_t threads[threads_num];
    thread_data_t thread_data[threads_num];
    
    uint32_t chunk = len / threads_num;
    uint32_t remainder = len % threads_num;
    uint32_t current_start = 0;
    
    for (int i = 0; i < threads_num; i++) {
        uint32_t chunk_size = chunk + (i < remainder ? 1 : 0);
        thread_data[i].id = i;
        thread_data[i].start = current_start;
        thread_data[i].end = current_start + chunk_size;
        thread_data[i].len = len;
        thread_data[i].A = A;
        thread_data[i].iteration = seed;
        thread_data[i].fixed = fixed;
        thread_data[i].M1 = *M1_p;
        thread_data[i].M2 = *M2_p;
        thread_data[i].iteration_idx = iteration_idx;
        pthread_create(&threads[i], NULL, generate_thread, &thread_data[i]);
        current_start += chunk_size;
    }
    
    for (int i = 0; i < threads_num; i++) {
        pthread_join(threads[i], NULL);
    }
    
    double end_time = get_wtime();
    pthread_mutex_lock(&timing_mutex);
    timing_stats[iteration_idx].generate_time += (end_time - start_time);
    pthread_mutex_unlock(&timing_mutex);
}

void *map_stage1_thread(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    for (uint32_t i = data->start; i < data->end; i++) {
        data->M1[i] = sinh(data->M1[i]);
        data->M1[i] = data->M1[i] * data->M1[i];
        if (i < data->len / 2) {
            data->M2[i] = log10(data->M2[i]);
            data->M2[i] = pow(data->M2[i], EULERS);
        }
    }
    return NULL;
}

void *map_stage2_thread(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    for (uint32_t i = data->start; i < data->end; i++) {
        data->M2_copy[i] = data->M2[i];
        data->M2[i] = 0;
    }
    return NULL;
}

void *map_stage3_thread(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    for (uint32_t i = data->start; i < data->end; i++) {
        for (uint32_t j = 0; j <= i; j++) {
            data->M2[i] += data->M2_copy[j];
        }
    }
    return NULL;
}

void map(float * const M1, float * const M2, const uint32_t len, int threads_num, int iteration_idx) {
    double start_time = get_wtime();
    
    pthread_t threads[threads_num];
    thread_data_t thread_data[threads_num];
    
    uint32_t chunk = len / threads_num;
    uint32_t remainder = len % threads_num;
    uint32_t current_start = 0;
    
    for (int i = 0; i < threads_num; i++) {
        uint32_t chunk_size = chunk + (i < remainder ? 1 : 0);
        thread_data[i].id = i;
        thread_data[i].start = current_start;
        thread_data[i].end = current_start + chunk_size;
        thread_data[i].len = len;
        thread_data[i].M1 = M1;
        thread_data[i].M2 = M2;
        thread_data[i].iteration_idx = iteration_idx;
        pthread_create(&threads[i], NULL, map_stage1_thread, &thread_data[i]);
        current_start += chunk_size;
    }
    
    for (int i = 0; i < threads_num; i++) {
        pthread_join(threads[i], NULL);
    }

    float *M2_copy = (float *)malloc(sizeof(float) * (len / 2));
    
    current_start = 0;
    chunk = (len / 2) / threads_num;
    remainder = (len / 2) % threads_num;
    
    for (int i = 0; i < threads_num; i++) {
        uint32_t chunk_size = chunk + (i < remainder ? 1 : 0);
        thread_data[i].start = current_start;
        thread_data[i].end = current_start + chunk_size;
        thread_data[i].M2_copy = M2_copy;
        pthread_create(&threads[i], NULL, map_stage2_thread, &thread_data[i]);
        current_start += chunk_size;
    }
    
    for (int i = 0; i < threads_num; i++) {
        pthread_join(threads[i], NULL);
    }
    
    current_start = 0;
    for (int i = 0; i < threads_num; i++) {
        uint32_t chunk_size = chunk + (i < remainder ? 1 : 0);
        thread_data[i].start = current_start;
        thread_data[i].end = current_start + chunk_size;
        pthread_create(&threads[i], NULL, map_stage3_thread, &thread_data[i]);
        current_start += chunk_size;
    }
    
    for (int i = 0; i < threads_num; i++) {
        pthread_join(threads[i], NULL);
    }

    free(M2_copy);
    
    double end_time = get_wtime();
    pthread_mutex_lock(&timing_mutex);
    timing_stats[iteration_idx].map_time += (end_time - start_time);
    pthread_mutex_unlock(&timing_mutex);
}

void *merge_thread(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    for (uint32_t i = data->start; i < data->end; i++) {
        data->M2[i] = data->M1[i] >= data->M2[i] ? data->M1[i] : data->M2[i];
    }
    return NULL;
}

void merge(float * const M1, float * const M2, const uint32_t len, int threads_num, int iteration_idx) {
    double start_time = get_wtime();
    
    pthread_t threads[threads_num];
    thread_data_t thread_data[threads_num];
    
    uint32_t chunk = (len / 2) / threads_num;
    uint32_t remainder = (len / 2) % threads_num;
    uint32_t current_start = 0;
    
    for (int i = 0; i < threads_num; i++) {
        uint32_t chunk_size = chunk + (i < remainder ? 1 : 0);
        thread_data[i].id = i;
        thread_data[i].start = current_start;
        thread_data[i].end = current_start + chunk_size;
        thread_data[i].len = len;
        thread_data[i].M1 = M1;
        thread_data[i].M2 = M2;
        thread_data[i].iteration_idx = iteration_idx;
        pthread_create(&threads[i], NULL, merge_thread, &thread_data[i]);
        current_start += chunk_size;
    }
    
    for (int i = 0; i < threads_num; i++) {
        pthread_join(threads[i], NULL);
    }
    
    double end_time = get_wtime();
    pthread_mutex_lock(&timing_mutex);
    timing_stats[iteration_idx].merge_time += (end_time - start_time);
    pthread_mutex_unlock(&timing_mutex);
}

void *sort_partial_thread(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    float tmp = 0;
    
    for (uint32_t j = data->start; j < data->end;) {
        if (j == data->start) {
            j++;
            continue;
        }
        if (data->M2[j] < data->M2[j-1]) {
            tmp = data->M2[j];
            data->M2[j] = data->M2[j-1];
            data->M2[j-1] = tmp;
            j--;
        } else {
            j++;
        }
    }
    return NULL;
}

void sort_list(float ** M2_p, const uint32_t len, int threads_num, int iteration_idx) {
    double start_time = get_wtime();
    
    uint32_t k = threads_num;
    uint32_t partial_len = ceil((len / 2.) / k);
    uint32_t last_part_len = (len / 2) - (k - 1) * partial_len;
    
    uint32_t *sizes = (uint32_t *)malloc(sizeof(uint32_t) * k);
    uint32_t *tips = (uint32_t *)malloc(sizeof(uint32_t) * k);
    uint32_t *shifts = (uint32_t *)malloc(sizeof(uint32_t) * k);
    float *new_M2 = (float *)malloc(sizeof(float) * len / 2);
    float *M2 = *M2_p;

    for (int i = 0; i < k; i++) {
        sizes[i] = i < (k-1) ? partial_len : last_part_len;
        tips[i] = 0;
        shifts[i] = partial_len * i;
    }
    
    pthread_t threads[k];
    thread_data_t thread_data[k];
    
    for (int i = 0; i < k; i++) {
        thread_data[i].id = i;
        thread_data[i].start = shifts[i];
        thread_data[i].end = shifts[i] + sizes[i];
        thread_data[i].M2 = M2;
        thread_data[i].iteration_idx = iteration_idx;
        pthread_create(&threads[i], NULL, sort_partial_thread, &thread_data[i]);
    }
    
    for (int i = 0; i < k; i++) {
        pthread_join(threads[i], NULL);
    }
    
    for (uint32_t i = 0; i < len / 2; i++) {
        uint32_t min_thread_idx = 0;
        float min = INFINITY;

        for (uint32_t j = 0; j < k; j++) {
            if (tips[j] == sizes[j]) continue;
            
            if (M2[tips[j] + shifts[j]] < min) {
                min_thread_idx = j;
                min = M2[tips[j] + shifts[j]];
            }
        }
        
        new_M2[i] = min;
        tips[min_thread_idx]++;
    }

    free(*M2_p);
    *M2_p = new_M2;
    free(sizes);
    free(tips);
    free(shifts);
    
    double end_time = get_wtime();
    pthread_mutex_lock(&timing_mutex);
    timing_stats[iteration_idx].sort_time += (end_time - start_time);
    pthread_mutex_unlock(&timing_mutex);
}

typedef struct {
    uint32_t id;
    uint32_t start;
    uint32_t end;
    float *M2;
    float compare;
    float partial_sum;
    int iteration_idx;
} reduce_thread_data_t;

void *reduce_thread(void *arg) {
    reduce_thread_data_t *data = (reduce_thread_data_t *)arg;
    data->partial_sum = 0;
    
    for (uint32_t i = data->start; i < data->end; i++) {
        if ((int)(data->M2[i] / data->compare) % 2 == 0) {
            float tmp = sin(data->M2[i]);
            if (!isnan(tmp)) {
                data->partial_sum += tmp;
            }
        }
    }
    return NULL;
}

float reduce(float * const M2, const uint32_t len, const int no_sort, int threads_num, int iteration_idx) {
    double start_time = get_wtime();
    
    float compare = 0;
    float sum = 0;

    compare = M2[0];
    if (no_sort) {
        for (uint32_t i = 0; i < len / 2; i++) {
            if ((fabsf(M2[i]) >= MINIMAL_DIVISOR) && (fabsf(M2[i]) < compare)) {
                compare = M2[i];
            }
        }
    }

    reduce_thread_data_t thread_data[threads_num];
    pthread_t threads[threads_num];
    
    uint32_t chunk = (len / 2) / threads_num;
    uint32_t remainder = (len / 2) % threads_num;
    uint32_t current_start = 0;
    
    for (int i = 0; i < threads_num; i++) {
        uint32_t chunk_size = chunk + (i < remainder ? 1 : 0);
        thread_data[i].id = i;
        thread_data[i].start = current_start;
        thread_data[i].end = current_start + chunk_size;
        thread_data[i].M2 = M2;
        thread_data[i].compare = compare;
        thread_data[i].iteration_idx = iteration_idx;
        pthread_create(&threads[i], NULL, reduce_thread, &thread_data[i]);
        current_start += chunk_size;
    }
    
    for (int i = 0; i < threads_num; i++) {
        pthread_join(threads[i], NULL);
        sum += thread_data[i].partial_sum;
    }
    
    double end_time = get_wtime();
    pthread_mutex_lock(&timing_mutex);
    timing_stats[iteration_idx].reduce_time += (end_time - start_time);
    pthread_mutex_unlock(&timing_mutex);

    return sum;
}

// Function to print timing statistics
void print_timing_stats(TimingStats *stats, int iterations, int write_to_file) {
    TimingStats totals = {0};
    FILE *csv_file = NULL;
    
    if (write_to_file) {
        csv_file = fopen("timing_results_pthreads.csv", "w");
        if (csv_file) {
            fprintf(csv_file, "iteration,total_time,generate_time,map_time,merge_time,sort_time,reduce_time,overhead_time\n");
        }
    }
    
    printf("\n=== Detailed Timing Statistics (Pthreads) ===\n");
    printf("Iteration |   Total  | Generate |    Map   |  Merge  |   Sort   |  Reduce  | Overhead\n");
    printf("----------|----------|----------|----------|----------|----------|----------|----------\n");
    
    for (int i = 0; i < iterations; i++) {
        TimingStats *ts = &stats[i];
        
        // Calculate overhead as difference between total and sum of components
        double components_sum = ts->generate_time + ts->map_time + ts->merge_time + 
                               ts->sort_time + ts->reduce_time;
        ts->overhead_time = ts->total_time - components_sum;
        
        printf("%9d | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f\n",
               i, ts->total_time * 1000.0, ts->generate_time * 1000.0,
               ts->map_time * 1000.0, ts->merge_time * 1000.0,
               ts->sort_time * 1000.0, ts->reduce_time * 1000.0,
               ts->overhead_time * 1000.0);
        
        // Accumulate totals
        totals.total_time += ts->total_time;
        totals.generate_time += ts->generate_time;
        totals.map_time += ts->map_time;
        totals.merge_time += ts->merge_time;
        totals.sort_time += ts->sort_time;
        totals.reduce_time += ts->reduce_time;
        totals.overhead_time += ts->overhead_time;
        
        // Write to CSV if requested
        if (csv_file) {
            fprintf(csv_file, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    i, ts->total_time, ts->generate_time, ts->map_time,
                    ts->merge_time, ts->sort_time, ts->reduce_time, ts->overhead_time);
        }
    }
    
    printf("----------|----------|----------|----------|----------|----------|----------|----------\n");
    printf("%9s | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f\n",
           "TOTAL", totals.total_time * 1000.0, totals.generate_time * 1000.0,
           totals.map_time * 1000.0, totals.merge_time * 1000.0,
           totals.sort_time * 1000.0, totals.reduce_time * 1000.0,
           totals.overhead_time * 1000.0);
    
    printf("%9s | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f\n",
           "AVG", (totals.total_time/iterations) * 1000.0, 
           (totals.generate_time/iterations) * 1000.0,
           (totals.map_time/iterations) * 1000.0,
           (totals.merge_time/iterations) * 1000.0,
           (totals.sort_time/iterations) * 1000.0,
           (totals.reduce_time/iterations) * 1000.0,
           (totals.overhead_time/iterations) * 1000.0);
    
    // Calculate percentages
    printf("\n=== Time Distribution (Percentage of Total) ===\n");
    printf("Generate: %.1f%%\n", (totals.generate_time / totals.total_time) * 100.0);
    printf("Map:      %.1f%%\n", (totals.map_time / totals.total_time) * 100.0);
    printf("Merge:    %.1f%%\n", (totals.merge_time / totals.total_time) * 100.0);
    printf("Sort:     %.1f%%\n", (totals.sort_time / totals.total_time) * 100.0);
    printf("Reduce:   %.1f%%\n", (totals.reduce_time / totals.total_time) * 100.0);
    printf("Overhead: %.1f%%\n", (totals.overhead_time / totals.total_time) * 100.0);
    
    if (csv_file) {
        fclose(csv_file);
        printf("\nDetailed timing data saved to 'timing_results_pthreads.csv'\n");
    }
}

void *progress_thread(void *arg) {
    double last_print = get_wtime();
    
    while (1) {
        pthread_mutex_lock(&progress_mutex);
        int tmp_finished = finished;
        int tmp_progress = progress;
        pthread_mutex_unlock(&progress_mutex);
        
        if (tmp_finished) {
            break;
        }
        
        if (get_wtime() - last_print >= 1.0) {
            printf("current progress: %i\n", tmp_progress);
            last_print = get_wtime();
        }
    }
    return NULL;
}

typedef struct {
    uint32_t N;
    int fixed_seq;
    int no_sort;
    int threads_num;
    int write_timing_csv;
    TimingStats *timing_stats;
} worker_args_t;

void *worker_thread(void *arg) {
    worker_args_t *args = (worker_args_t *)arg;
    uint32_t N = args->N;
    int fixed_seq = args->fixed_seq;
    int no_sort = args->no_sort;
    int threads_num = args->threads_num;
    TimingStats *timing_stats = args->timing_stats;
    uint32_t A = 256;
    
    const int ITERATIONS = 100;
    
    for (uint32_t i = 0; i < ITERATIONS; i++) {
        double iteration_start = get_wtime();
        
        pthread_mutex_lock(&timing_mutex);
        current_iteration = i;
        timing_stats[i].iteration_count = i;
        pthread_mutex_unlock(&timing_mutex);
        
        pthread_mutex_lock(&progress_mutex);
        progress = i;
        pthread_mutex_unlock(&progress_mutex);
        
        uint32_t seed = i;
        srand(i);
        float *M1 = 0;
        float *M2 = 0;

        generate(&M1, &M2, N, A, seed, fixed_seq, threads_num, i);
        map(M1, M2, N, threads_num, i);
        merge(M1, M2, N, threads_num, i);
        if (!no_sort) {
            sort_list(&M2, N, threads_num, i);
        } else {
            pthread_mutex_lock(&timing_mutex);
            timing_stats[i].sort_time = 0.0;
            pthread_mutex_unlock(&timing_mutex);
        }
        float iteration_result = reduce(M2, N, no_sort, threads_num, i);

        free(M1);
        free(M2);
        
        double iteration_end = get_wtime();
        pthread_mutex_lock(&timing_mutex);
        timing_stats[i].total_time = iteration_end - iteration_start;
        pthread_mutex_unlock(&timing_mutex);
        
        // Optional: Print per-iteration timing
        // printf("Iteration %d: %.3f ms\n", i, timing_stats[i].total_time * 1000.0);
    }
    
    pthread_mutex_lock(&progress_mutex);
    finished = 1;
    pthread_mutex_unlock(&progress_mutex);
    
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Function expects at least 2 arguments\n");
        return -1;
    }

    const int fixed_seq = atoi(argv[2]) > 2;
    const int no_sort = atoi(argv[2]) % 2 == 0;
    const int threads_num = argc > 3 ? atoi(argv[3]) : 1;
    const int write_timing_csv = argc > 4 ? atoi(argv[4]) : 0;

    const uint32_t A = 256;
    int32_t N;
    double T1, T2;
    int64_t delta_ms;
    
    const int ITERATIONS = 100;

    printf("set threads num to %i\n", threads_num);

    N = atoi(argv[1]);
    
    // Allocate and initialize timing statistics
    timing_stats = (TimingStats *)malloc(ITERATIONS * sizeof(TimingStats));
    if (timing_stats == NULL) {
        fprintf(stderr, "Failed to allocate timing statistics\n");
        return -1;
    }
    
    for (int i = 0; i < ITERATIONS; i++) {
        timing_stats[i].total_time = 0.0;
        timing_stats[i].generate_time = 0.0;
        timing_stats[i].map_time = 0.0;
        timing_stats[i].merge_time = 0.0;
        timing_stats[i].sort_time = 0.0;
        timing_stats[i].reduce_time = 0.0;
        timing_stats[i].overhead_time = 0.0;
        timing_stats[i].iteration_count = 0;
    }
    
    T1 = get_wtime();
    
    // Initialize mutexes
    pthread_mutex_init(&progress_mutex, NULL);
    pthread_mutex_init(&compare_mutex, NULL);
    pthread_mutex_init(&timing_mutex, NULL);
    
    // Create worker arguments
    worker_args_t worker_args = {
        .N = N,
        .fixed_seq = fixed_seq,
        .no_sort = no_sort,
        .threads_num = threads_num,
        .write_timing_csv = write_timing_csv,
        .timing_stats = timing_stats
    };
    
    // Create worker and reporter threads
    pthread_t worker, reporter;
    pthread_create(&worker, NULL, worker_thread, &worker_args);
    pthread_create(&reporter, NULL, progress_thread, NULL);
    
    pthread_join(worker, NULL);
    pthread_join(reporter, NULL);
    
    // Destroy mutexes
    pthread_mutex_destroy(&progress_mutex);
    pthread_mutex_destroy(&compare_mutex);
    pthread_mutex_destroy(&timing_mutex);

    T2 = get_wtime();
    delta_ms = (T2 - T1) * 1000;
    
    printf("\nN=%d. Total milliseconds passed: %ld\n", N, delta_ms);
    printf("Average time per iteration: %.3f ms\n", delta_ms / (double)ITERATIONS);
    
    // Print detailed timing statistics
    print_timing_stats(timing_stats, ITERATIONS, write_timing_csv);
    
    // Free timing statistics
    free(timing_stats);
    
    return 0;
}