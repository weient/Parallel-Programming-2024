#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>


#define ull unsigned long long


typedef struct thread_data {
    ull start;
    ull end;
    ull r;
    ull k;
    ull result;
} t_data;

void *cal_pixels(void *arg) {
    t_data* data = (t_data*)arg;
    ull pixels = 0;

    for (ull x = data->start; x < data->end; x++) {
        ull y = ceil(sqrtl(data->r * data->r - x * x));
        pixels += y;
        pixels %= data->k;
    }

    data->result = pixels;
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }
    cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	ull NUM_THREADS = CPU_COUNT(&cpuset);
    //int NUM_THREADS = atoi(getenv("SLURM_CPUS_PER_TASK"));
    //printf("number of threads: %d\n", num_threads);
    ull r = atoll(argv[1]);
    ull k = atoll(argv[2]);
    ull total_pixels = 0;

    pthread_t threads[NUM_THREADS];
    t_data data_array[NUM_THREADS];

    ull job_per_th = r / NUM_THREADS;
    int remainder = r % NUM_THREADS;


    for (int i = 0; i < NUM_THREADS; i++) {
        data_array[i].r = r;
        data_array[i].k = k;
        data_array[i].start = i < remainder ? i * (job_per_th + 1) : remainder * (job_per_th + 1) + (i - remainder) * job_per_th;
        data_array[i].end = i < remainder ? data_array[i].start + job_per_th + 1 : data_array[i].start + job_per_th;
        data_array[i].result = 0;

        pthread_create(&threads[i], NULL, cal_pixels, (void *)&data_array[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        total_pixels += data_array[i].result;
        total_pixels %= k;
    }

    printf("%llu\n", (4 * total_pixels) % k);

    return 0;
}