#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#define ull unsigned long long
int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }
    ull r = atoll(argv[1]);
    ull k = atoll(argv[2]);
    ull global = 0;
    omp_lock_t lock;
    omp_init_lock(&lock);
    
    #pragma omp parallel
    {
        ull local = 0;
        int omp_threads = omp_get_num_threads();
        ull chunk = r / omp_threads;
        #pragma omp for schedule(guided, chunk) nowait
        for (ull x = 0; x < r; x++) {
            ull y = ceil(sqrtl(r*r - x*x));
            local += y;
            local %= k;
        }

        omp_set_lock(&lock);
        global += local;
        global %= k;
        omp_unset_lock(&lock);
    }
    
    omp_destroy_lock(&lock);
    printf("%llu\n", (4 * global) % k);
    return 0;
}