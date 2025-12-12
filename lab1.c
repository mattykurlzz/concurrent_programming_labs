#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

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
    
    if (fixed) {
        (*M1_p)[0] = 1;
    }

    for (uint32_t i = 0; i < len; i++) {
        if (fixed) {
            (*M1_p)[i] = (*M1_p)[0];  //(i % A) + 1;
        } else {
            (*M1_p)[i] = (rand_r(seed) % A) + 1;
        }
    }

    for (uint32_t i = 0; i < len / 2; i++) {
        if (fixed) {
            (*M2_p)[i] = (*M1_p)[0] * A;  //(i % (9 * A + 1)) + A;
        } else {
            (*M2_p)[i] = (rand_r(seed) % (9 * A + 1)) + A;
        }
    }
}

void map(float * const M1, float * const M2, const uint32_t len) {
    for (uint32_t i = 0; i < len; i++) {
        M1[i] = sinh(M1[i]);
        M1[i] = M1[i] * M1[i];
    }

    for (uint32_t i = 0; i < len / 2; i++) {
        M2[i] = log10(M2[i]);
        M2[i] = pow(M2[i], EULERS);
    }
}

void merge(float * const M1, float * const M2, const uint32_t len) {
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

float reduce(float * const M2, const uint32_t len) {
    float compare = 0;
    float sum = 0;
    float tmp = 0;

    compare = M2[0];
    for (uint32_t i = 0; i < len / 2; i++) {
        if ((fabsf(M2[i]) >= MINIMAL_DIVISOR) && (fabsf(M2[i]) < compare)) {
            compare = M2[i];
        }
    }

    for (uint32_t i = 0; i < len / 2; i++) {
        if ((int)(M2[i] / compare) % 2 == 0) {
            tmp = sin(M2[i]);
            if (!isnan(tmp)) {
                sum += sin(M2[i]);
            }
        }
    }

    return sum;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Function expects at least 2 arguments\n");
        return -1;
    }
    const int fixed_seq = atoi(argv[2]) > 2;
    const int no_sort = atoi(argv[2]) % 2 == 0;

    const uint32_t A = 256;
    int32_t N;
    float iteration_result = 0;
    struct timeval T1, T2;
    int64_t delta_ms;

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
        iteration_result = reduce(M2, N);

        printf("reduce result: %f\n", iteration_result);
    }
    gettimeofday(&T2, NULL);
    delta_ms = (T2.tv_sec - T1.tv_sec) * 1000 +
        (T2.tv_usec - T1.tv_usec) / 1000;
    printf("\nN=%d. Milliseconds passed: %ld\n", N, delta_ms);
    return 0;
}
