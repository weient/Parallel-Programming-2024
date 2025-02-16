#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#define ull unsigned long long
#define CAL(r, x) ceil(sqrtl(sq(r) - sq(x)))
#define MIN(a, b) (((a)<(b))?(a):(b))


ull sq(ull n) {
 
    // Initialize result
    ull res = n;
 
    // Add n to res n-1 times
    for (ull i = 1; i < n; i++)
        res += n;
 
    return res;
}


ull modulo(ull a, ull b) {
    
    if (a < b) return a;
    
    ull M = modulo(a, b << 1);
    
    if (M >= b) M -= b;
	
    return M;
}
int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	ull r = atoll(argv[1]);
	ull k = atoll(argv[2]);
	ull pixels = 0;


	ull max_cpu_used;
	int rank, size;
    MPI_Status stat;
	int flag = 0;
	//double starttime, endtime;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//starttime = MPI_Wtime();
	max_cpu_used = MIN(r, size);

	for (ull x = rank; x < r; x+=size)
	{
		ull y = CAL(r, x);
		pixels += y; 
		pixels = modulo(pixels, k);
		//printf("after mod %llu\n", pixels);
		//pixels %= k;
	}
	//endtime   = MPI_Wtime();
	//printf("rank %d took %f seconds\n",rank, endtime-starttime);
	ull total_pixels = 0;
	if (rank == 0)
	{
		MPI_Reduce(&pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
		total_pixels = modulo(total_pixels, k);
        //total_pixels %= k;
        printf("%llu\n", modulo((total_pixels << 2) , k));
	}
	else if (rank < max_cpu_used) MPI_Reduce(&pixels, NULL, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	
	//endtime   = MPI_Wtime();
	//printf("rank %d took %f seconds\n",rank, endtime-starttime);

	MPI_Finalize();
}
