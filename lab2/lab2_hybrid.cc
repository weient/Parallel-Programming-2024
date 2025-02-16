#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#define ull unsigned long long

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }
    ull r = atoll(argv[1]);
    ull k = atoll(argv[2]);
    ull local = 0, global = 0;
    ull start, end;

    ull chunk = r / size;
    ull remainder = r % size;

    start = rank < remainder ? rank * (chunk + 1) : remainder * (chunk + 1) + (rank - remainder) * chunk;
    end = rank < remainder ? start + chunk + 1 : start + chunk;
    omp_lock_t lock;
    omp_init_lock(&lock);

    #pragma omp parallel
    {
        ull t_local = 0;
        #pragma omp for nowait
        for (ull x = start; x < end; x++) {
            ull y = ceil(sqrtl(r*r - x*x));
            t_local += y;
            t_local %= k;
        }
        omp_set_lock(&lock);
        local += t_local;
        local %= k;
        omp_unset_lock(&lock);
    }

    omp_destroy_lock(&lock);
    MPI_Reduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        global %= k;
        printf("%llu\n", (4 * global) % k);
    }

    MPI_Finalize();
    return 0;
}