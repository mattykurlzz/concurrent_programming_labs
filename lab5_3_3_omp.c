#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#else
double omp_get_wtime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}
#endif

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

// Global timing structure (one per iteration)
TimingStats *timing_stats = NULL;
int current_iteration = 0;

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

void generate(float **M1_p, float **M2_p, const uint32_t len, const uint32_t A,
              unsigned int seed, const int fixed) {
    double start_time = omp_get_wtime();
    
    *M1_p = (float *)malloc(sizeof(float) * len);
    *M2_p = (float *)malloc(sizeof(float) * (len / 2));

    if (*M1_p == NULL || *M2_p == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    // we need to introduce variance because either way results are always 0
    if (fixed && (len > 1)) {
        (*M1_p)[0] = 1.2;
        (*M1_p)[1] = 3;
    }

    uint32_t iteration = seed;

    #pragma omp parallel for default(none) shared(M1_p, M2_p, len, fixed, A, iteration) schedule(static, 8)
    for (uint32_t i = 0; i < len; i++) {
        if (fixed) {
            (*M1_p)[i] = (*M1_p)[i % 2];
            if (i < len / 2) {
                (*M2_p)[i] = (*M1_p)[i % 2] * A;
            }
        } else {
            // generate seed based on position and iteration
            unsigned int local_seed = hash_position(i, iteration);
            
            (*M1_p)[i] = (rand_r(&local_seed) % A) + 1;
            
            if (i < len / 2) {
                local_seed = hash_position(i + 0x10000, iteration);
                (*M2_p)[i] = (rand_r(&local_seed) % (9 * A + 1)) + A;
            }
        }
    }
    
    double end_time = omp_get_wtime();
    if (timing_stats != NULL) {
        timing_stats[current_iteration].generate_time += (end_time - start_time);
    }
}

void map(float * const M1, float * const M2, const uint32_t len) {
    double start_time = omp_get_wtime();
    
    #pragma omp parallel for default(none) shared(M1, M2, len) schedule(guided, 16)
    for (uint32_t i = 0; i < len; i++) {
        M1[i] = sinh(M1[i]);
        M1[i] = M1[i] * M1[i];
        if (i < len / 2) {
            M2[i] = log10(M2[i]);
            M2[i] = pow(M2[i], EULERS);
        }
    }

    float *M2_copy = (float *)malloc(sizeof(float) * (len / 2));

    #pragma omp parallel for default(none) shared(M1, M2, M2_copy, len) schedule(static, 17)
    for (size_t i = 0; i < len / 2; i++) {
      M2_copy[i] = M2[i];
      M2[i] = 0;
    }

    #pragma omp parallel for default(none) shared(M2, M2_copy, len) schedule(dynamic, 16)
    for (size_t i = 0; i < len / 2; i++) {
        for (size_t j = 0; j <= i; j++) {
          M2[i] += M2_copy[j];
        }
    }

    free(M2_copy);
    
    double end_time = omp_get_wtime();
    if (timing_stats != NULL) {
        timing_stats[current_iteration].map_time += (end_time - start_time);
    }
}

void merge(float * const M1, float * const M2, const uint32_t len) {
    double start_time = omp_get_wtime();
    
    #pragma omp parallel for default(none) shared(len, M1, M2) schedule(guided, 3)
    for (uint32_t i = 0; i < len / 2; i++) {
        M2[i] = M1[i] >= M2[i] ? M1[i] : M2[i];
    }
    
    double end_time = omp_get_wtime();
    if (timing_stats != NULL) {
        timing_stats[current_iteration].merge_time += (end_time - start_time);
    }
}

void sort_list(float ** M2_p, const uint32_t len) {
    double start_time = omp_get_wtime();
    
    uint32_t k = omp_get_max_threads();
    
    uint32_t partial_len = ceil((len / 2.) / k);
    uint32_t last_part_len = (len / 2) - (k - 1) * partial_len;
    
    uint32_t *sizes = (uint32_t *)malloc(sizeof(uint32_t) * k);
    uint32_t *tips = (uint32_t *)malloc(sizeof(uint32_t) * k);
    uint32_t *shifts = (uint32_t *)malloc(sizeof(uint32_t) * k);
    float *new_M2 = (float *)malloc(sizeof(float) * len / 2);
    float *M2 = *M2_p;

    for (int i = 0; i < k; i++ ) {
        sizes[i] = i < (k-1) ? partial_len : last_part_len;
        tips[i] = 0;
        shifts[i] = partial_len * i;
    }
    
    #pragma omp parallel num_threads(k)
    {
        uint32_t current_thread = omp_get_thread_num();
        float tmp = 0;

        for (uint32_t j = shifts[current_thread]; j < shifts[current_thread] + sizes[current_thread];) {
            if ( j == shifts[current_thread] ) {
                j++;
                continue;
            }
            if (M2[j] < M2[j-1]) {
                tmp = M2[j];
                M2[j] = M2[j-1];
                M2[j-1] = tmp;
                
                j--;
            } else {
                j++;
            }
        }
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
    printf("\n");

    free(*M2_p);
    
    *M2_p = new_M2;
    
    free(sizes);
    free(tips);
    free(shifts);
    
    double end_time = omp_get_wtime();
    if (timing_stats != NULL) {
        timing_stats[current_iteration].sort_time += (end_time - start_time);
    }
}

float reduce(float * const M2, const uint32_t len, const int no_sort) {
    double start_time = omp_get_wtime();
    
    float compare = 0;
    float sum = 0;
    float tmp = 0;

    compare = M2[0];
    // concurrent reading and writing to compare may break cycle logic
    if (no_sort) {
        for (uint32_t i = 0; i < len / 2; i++) {
            if ((fabsf(M2[i]) >= MINIMAL_DIVISOR) && (fabsf(M2[i]) < compare)) {
                compare = M2[i];
            }
        }
    }

    #pragma omp parallel for default(none) private(tmp) shared(len, M2, compare) schedule(guided, 13) reduction(+:sum)
    for (uint32_t i = 0; i < len / 2; i++) {
        if ((int)(M2[i] / compare) % 2 == 0) {
            tmp = sin(M2[i]);
            if (!isnan(tmp)) {
                sum += tmp;
            }
        }
    }
    
    double end_time = omp_get_wtime();
    if (timing_stats != NULL) {
        timing_stats[current_iteration].reduce_time += (end_time - start_time);
    }

    return sum;
}

// Function to print timing statistics
void print_timing_stats(int iterations, int write_to_file) {
    TimingStats totals = {0};
    FILE *csv_file = NULL;
    
    if (write_to_file) {
        csv_file = fopen("timing_results.csv", "w");
        if (csv_file) {
            fprintf(csv_file, "iteration,total_time,generate_time,map_time,merge_time,sort_time,reduce_time,overhead_time\n");
        }
    }
    
    printf("\n=== Detailed Timing Statistics ===\n");
    printf("Iteration |   Total  | Generate |    Map   |  Merge  |   Sort   |  Reduce  | Overhead\n");
    printf("----------|----------|----------|----------|----------|----------|----------|----------\n");
    
    for (int i = 0; i < iterations; i++) {
        TimingStats *ts = &timing_stats[i];
        
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
        printf("\nDetailed timing data saved to 'timing_results.csv'\n");
    }
}

// build command: gcc -O3 -Wall -fopenmp -o lab3_parallel_with_fopenmp lab1.c -lm
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Function expects at least 2 arguments\n");
        return -1;
    }

    const int fixed_seq = atoi(argv[2]) > 2; // T if 3 or 4
    const int no_sort = atoi(argv[2]) % 2 == 0; // T if 2 or 4
    const int threads_num = argc > 3 ? atoi(argv[3]) : 1;
    const int write_timing_csv = argc > 4 ? atoi(argv[4]) : 0;

    int finished = 0;

    const uint32_t A = 256;
    int32_t N;
    float iteration_result = 0;
    double T1, T2, Ttmp;
    int64_t delta_ms;
    
    const int ITERATIONS = 100;

#ifdef _OPENMP
    printf("set threads num to %i\n", threads_num);
    omp_set_num_threads(threads_num);
#endif

    N = atoi(argv[1]);
    
    // Allocate timing statistics
    timing_stats = (TimingStats *)malloc(ITERATIONS * sizeof(TimingStats));
    if (timing_stats == NULL) {
        fprintf(stderr, "Failed to allocate timing statistics\n");
        return -1;
    }
    
    // Initialize timing stats
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
    
    T1 = Ttmp = omp_get_wtime();
    omp_set_nested(1); 
    
    uint32_t progress = 0;
    #pragma omp parallel sections shared(finished, progress, current_iteration) num_threads(threads_num)
    {
        #pragma omp section
        {
            for (uint32_t i = 0; i < ITERATIONS; i++) {
                double iteration_start = omp_get_wtime();
                current_iteration = i;
                timing_stats[i].iteration_count = i;
                
                #pragma omp atomic write
                progress = i;

                uint32_t seed = i;
                // printf("iteration %u\n", i);

                srand(i);
                float *M1 = 0;
                float *M2 = 0;

                generate(&M1, &M2, N, A, seed, fixed_seq);
                map(M1, M2, N);
                merge(M1, M2, N);
                if (!no_sort) {
                    sort_list(&M2, N);
                } else {
                    timing_stats[i].sort_time = 0.0;
                }
                iteration_result = reduce(M2, N, no_sort);

                free(M1);
                free(M2);

                double iteration_end = omp_get_wtime();
                timing_stats[i].total_time = iteration_end - iteration_start;
                
                // Optional: Print per-iteration timing
                // printf("Iteration %d: %.3f ms\n", i, timing_stats[i].total_time * 1000.0);
            }
            #pragma omp atomic write
            finished = 1;
        }
#ifdef _OPENMP
        #pragma omp section
        {
            while (!finished) {
                int tmp_finished = 0;
                #pragma omp atomic read
                tmp_finished = finished;
                if (tmp_finished) {
                    break;
                }
                if(Ttmp < omp_get_wtime() - 1) {
                    int tmp_progress = 0;
                    #pragma omp atomic read
                    tmp_progress = progress;
                    printf("current progress: %i\n", tmp_progress);
                    Ttmp = omp_get_wtime();
                }
            }
        }
#endif
    }

    T2 = omp_get_wtime();
    delta_ms = (T2 - T1) * 1000;
    
    printf("\nN=%d. Total milliseconds passed: %ld\n", N, delta_ms);
    printf("Average time per iteration: %.3f ms\n", delta_ms / (double)ITERATIONS);
    
    // Print detailed timing statistics
    print_timing_stats(ITERATIONS, write_timing_csv);
    
    // Free timing statistics
    free(timing_stats);
    
    return 0;
}