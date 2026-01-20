#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>

#define EULERS (2.718281828459045)
#define MINIMAL_DIVISOR 0.001

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
} thread_data_t;

pthread_mutex_t progress_mutex;
pthread_mutex_t compare_mutex;
int progress = 0;
int finished = 0;
float global_compare = 0;

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
              unsigned int seed, const int fixed, int threads_num) {
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
        pthread_create(&threads[i], NULL, generate_thread, &thread_data[i]);
        current_start += chunk_size;
    }
    
    for (int i = 0; i < threads_num; i++) {
        pthread_join(threads[i], NULL);
    }
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

void map(float * const M1, float * const M2, const uint32_t len, int threads_num) {
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
}

void *merge_thread(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    for (uint32_t i = data->start; i < data->end; i++) {
        data->M2[i] = data->M1[i] >= data->M2[i] ? data->M1[i] : data->M2[i];
    }
    return NULL;
}

void merge(float * const M1, float * const M2, const uint32_t len, int threads_num) {
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
        pthread_create(&threads[i], NULL, merge_thread, &thread_data[i]);
        current_start += chunk_size;
    }
    
    for (int i = 0; i < threads_num; i++) {
        pthread_join(threads[i], NULL);
    }
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

void sort_list(float ** M2_p, const uint32_t len, int threads_num) {
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
}

typedef struct {
    uint32_t id;
    uint32_t start;
    uint32_t end;
    float *M2;
    float compare;
    float partial_sum;
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

float reduce(float * const M2, const uint32_t len, const int no_sort, int threads_num) {
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
        pthread_create(&threads[i], NULL, reduce_thread, &thread_data[i]);
        current_start += chunk_size;
    }
    
    for (int i = 0; i < threads_num; i++) {
        pthread_join(threads[i], NULL);
        sum += thread_data[i].partial_sum;
    }

    return sum;
}

double get_wtime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
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

void *worker_thread(void *arg) {
    int *params = (int *)arg;
    uint32_t N = params[0];
    int fixed_seq = params[1];
    int no_sort = params[2];
    int threads_num = params[3];
    uint32_t A = 256;
    
    for (uint32_t i = 0; i < 100; i++) {
        pthread_mutex_lock(&progress_mutex);
        progress = i;
        pthread_mutex_unlock(&progress_mutex);
        
        uint32_t seed = i;
        srand(i);
        float *M1 = 0;
        float *M2 = 0;

        generate(&M1, &M2, N, A, seed, fixed_seq, threads_num);
        map(M1, M2, N, threads_num);
        merge(M1, M2, N, threads_num);
        if (!no_sort) {
            sort_list(&M2, N, threads_num);
        }
        float iteration_result = reduce(M2, N, no_sort, threads_num);

        free(M1);
        free(M2);
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

    const uint32_t A = 256;
    int32_t N;
    double T1, T2;
    int64_t delta_ms;

    printf("set threads num to %i\n", threads_num);

    N = atoi(argv[1]);
    T1 = get_wtime();
    
    pthread_mutex_init(&progress_mutex, NULL);
    pthread_mutex_init(&compare_mutex, NULL);
    
    int params[4] = {N, fixed_seq, no_sort, threads_num};
    
    pthread_t worker, reporter;
    pthread_create(&worker, NULL, worker_thread, params);
    pthread_create(&reporter, NULL, progress_thread, NULL);
    
    pthread_join(worker, NULL);
    pthread_join(reporter, NULL);
    
    pthread_mutex_destroy(&progress_mutex);
    pthread_mutex_destroy(&compare_mutex);

    T2 = get_wtime();
    delta_ms = (T2 - T1) * 1000;
    printf("\nN=%d. Milliseconds passed: %ld\n", N, delta_ms);
    return 0;
}