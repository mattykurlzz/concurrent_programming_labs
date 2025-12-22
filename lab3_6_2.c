#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>

#define EULERS (2.718281828459045)
#define MINIMAL_DIVISOR 0.001

void generate(float **M1_p, float **M2_p, const uint32_t len, const uint32_t A,
              unsigned int *seed, const int fixed) {
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

    // parallelization screws up random values because of unordered seed modification
    #pragma omp parallel for default(none) shared(M1_p, M2_p, len, fixed, A, seed) schedule(static, 8)
    for (uint32_t i = 0; i < len; i++) {
        unsigned int local_seed = seed + i;
        if (fixed) {
            (*M1_p)[i] = (*M1_p)[i % 2];  //(i % A) + 1;
            if ( i < len / 2) {
                (*M2_p)[i] = (*M1_p)[i % 2] * A;  //(i % (9 * A + 1)) + A;
            }
        } else {
            (*M1_p)[i] = (rand_r(&local_seed) % A) + 1;
            if ( i < len / 2) {
                (*M2_p)[i] = (rand_r(&local_seed) % (9 * A + 1)) + A;
            }
        }
    }
}

void map(float * const M1, float * const M2, const uint32_t len) {
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
}

void merge(float * const M1, float * const M2, const uint32_t len) {
    #pragma omp parallel for default(none) shared(len, M1, M2) schedule(guided, 3)
    for (uint32_t i = 0; i < len / 2; i++) {
        M2[i] = M1[i] >= M2[i] ? M1[i] : M2[i];
    }
}

void sort_list(float * const M2, const uint32_t len) {
    float tmp = 0;
    uint32_t i = 1;
    while (i < len / 2) {
        if (i == 0 || M2[i] >= M2[i-1]) {
            i++;
        } else {
            tmp = M2[i];
            M2[i] = M2[i-1];
            M2[i-1] = tmp;
            i--;
        }
    }
}

float reduce(float * const M2, const uint32_t len, const int no_sort) {
    float compare = 0;
    float sum = 0;
    float tmp = 0;

    compare = M2[0];
    // cocncurrent reading and writing to compare may break cycle logic
    if (no_sort) {
        for (uint32_t i = 0; i < len / 2; i++) {
            if ((fabsf(M2[i]) >= MINIMAL_DIVISOR) && (fabsf(M2[i]) < compare)) {
                compare = M2[i];
            }
        }
    }

    #pragma omp parallel for default(none) private(tmp) shared(len, M2, compare, sum) schedule(guided, 13) reduction(+:sum)
    for (uint32_t i = 0; i < len / 2; i++) {
        if ((int)(M2[i] / compare) % 2 == 0) {
            tmp = sin(M2[i]);
            if (!isnan(tmp)) {
                sum += tmp;
            }
        }
    }

    return sum;
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

    const uint32_t A = 256;
    int32_t N;
    float iteration_result = 0;
    struct timeval T1, T2;
    int64_t delta_ms;

#ifdef _OPENMP
    printf("set threads num to %i\n", threads_num);
    omp_set_num_threads(threads_num);
#endif

    N = atoi(argv[1]);
    gettimeofday(&T1, NULL);
    for (uint32_t i = 0; i < 100; i++) {
        uint32_t seed = i;
        printf("iteration %u\n", i);

        srand(i);
        float *M1 = 0;
        float *M2 = 0;

        generate(&M1, &M2, N, A, &seed, fixed_seq);
        map(M1, M2, N);
        merge(M1, M2, N);
        if (!no_sort) {
            sort_list(M2, N);
        }
        iteration_result = reduce(M2, N, no_sort);

        free(M1);
        free(M2);

        printf("reduce result: %f\n", iteration_result);
    }
    gettimeofday(&T2, NULL);
    delta_ms = (T2.tv_sec - T1.tv_sec) * 1000 +
        (T2.tv_usec - T1.tv_usec) / 1000;
    printf("\nN=%d. Milliseconds passed: %ld\n", N, delta_ms);
    return 0;
}
