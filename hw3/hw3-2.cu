#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define TS 32
#define BS 64
const int INF = ((1 << 30) - 1);
int n, m, n_ori;

int ceil_div(int a, int b) { 
    return (a + b - 1) / b; 
}
__global__ void phase1_kernel(int* d_Dist, int n, int Round) {
    int half = BS >> 1;
    int il = threadIdx.x;
    int jl = threadIdx.y;
    int p = Round * BS;
    int i = p + il;
    int j = p + jl;
    
    int i_n = i * n;
    int i_half_n = (i + half) * n;

    __shared__ int shared[BS][BS + 1];
    shared[il][jl] = d_Dist[i_n + j];
    shared[il+half][jl] = d_Dist[i_half_n + j];
    shared[il][jl+half] = d_Dist[i_n + (j+half)];
    shared[il+half][jl+half] = d_Dist[i_half_n + (j+half)];
    

    #pragma unroll
    for (int k = 0; k < BS; k++) {
        __syncthreads();
        shared[il][jl] = min(shared[il][jl], shared[il][k] + shared[k][jl]);
        shared[il+half][jl] = min(shared[il+half][jl], shared[il+half][k] + shared[k][jl]);
        shared[il][jl+half] = min(shared[il][jl+half], shared[il][k] + shared[k][jl+half]);
        shared[il+half][jl+half] = min(shared[il+half][jl+half], shared[il+half][k] + shared[k][jl+half]);

    }
    d_Dist[i_n + j] = shared[il][jl];
    d_Dist[i_half_n + j] = shared[il+half][jl];
    d_Dist[i_n + (j+half)] = shared[il][jl+half];
    d_Dist[i_half_n + (j+half)] = shared[il+half][jl+half];
}



__global__ void phase2_kernel(int* d_Dist, int n, int Round) {
    int half = BS >> 1;
    int p = Round * BS;
    int il = threadIdx.y;
    int jl = threadIdx.x;
    int bid = blockIdx.x;
    if (bid >= Round) bid++;
    int is_row = blockIdx.y;

    

    int i = (is_row * p) + (!is_row * (bid * BS)) + il;
    int j = (!is_row * p) + (is_row * (bid * BS)) + jl;
    int i_n = i * n;
    int i_half_n = (i + half) * n;


    __shared__ int shared_p[BS][BS + 1];
    __shared__ int shared[BS][BS + 1];
    
    shared[il][jl] = d_Dist[i_n + j];
    shared[il+half][jl] = d_Dist[i_half_n + j];
    shared[il][jl+half] = d_Dist[i_n + (j+half)];
    shared[il+half][jl+half] = d_Dist[i_half_n + (j+half)];
    
    shared_p[il][jl] = d_Dist[(p + il) * n + (p + jl)];
    shared_p[il+half][jl] = d_Dist[(p + (il+half)) * n + (p + jl)];
    shared_p[il][jl+half] = d_Dist[(p + il) * n + (p + (jl+half))];
    shared_p[il+half][jl+half] = d_Dist[(p + (il+half)) * n + (p + (jl+half))];

    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BS; k++) {
        if (is_row) {
            shared[il][jl] = min(shared[il][jl], shared_p[il][k] + shared[k][jl]);
            shared[il+half][jl] = min(shared[il+half][jl], shared_p[il+half][k] + shared[k][jl]);
            shared[il][jl+half] = min(shared[il][jl+half], shared_p[il][k] + shared[k][jl+half]);
            shared[il+half][jl+half] = min(shared[il+half][jl+half], shared_p[il+half][k] + shared[k][jl+half]);
        } else {
            shared[il][jl] = min(shared[il][jl], shared[il][k] + shared_p[k][jl]);
            shared[il+half][jl] = min(shared[il+half][jl], shared[il+half][k] + shared_p[k][jl]);
            shared[il][jl+half] = min(shared[il][jl+half], shared[il][k] + shared_p[k][jl+half]);
            shared[il+half][jl+half] = min(shared[il+half][jl+half], shared[il+half][k] + shared_p[k][jl+half]);
        }
    }
    
    d_Dist[i_n + j] = shared[il][jl];
    d_Dist[i_half_n + j] = shared[il+half][jl];
    d_Dist[i_n + (j+half)] = shared[il][jl+half];
    d_Dist[i_half_n + (j+half)] = shared[il+half][jl+half];
}
__global__ void phase3_kernel(int* d_Dist, int n, int Round) {
    int half = BS >> 1;
    int p = Round * BS;
    int il = threadIdx.y;
    int jl = threadIdx.x;
    int blk_x = blockIdx.x + (blockIdx.x >= Round);
    int blk_y = blockIdx.y + (blockIdx.y >= Round);
    

    int i = blk_x * BS + il;
    int j = blk_y * BS + jl;
    int i_n = i * n;
    int i_half_n = (i + half) * n;

    __shared__ int shared[2 * BS * BS];  // 2 * 64 * 64

    int row_offset = 0;
    int col_offset = BS * BS;

    shared[row_offset + il * BS + jl] = d_Dist[i_n + (p + jl)];
    shared[row_offset + (il+half) * BS + jl] = d_Dist[i_half_n + (p + jl)];
    shared[row_offset + il * BS + (jl+half)] = d_Dist[i_n + (p + (jl+half))];
    shared[row_offset + (il+half) * BS + (jl+half)] = d_Dist[i_half_n + (p + (jl+half))];

    shared[col_offset + il * BS + jl] = d_Dist[(p + il) * n + j];
    shared[col_offset + (il+half) * BS + jl] = d_Dist[(p + (il+half)) * n + j];
    shared[col_offset + il * BS + (jl+half)] = d_Dist[(p + il) * n + (j+half)];
    shared[col_offset + (il+half) * BS + (jl+half)] = d_Dist[(p + (il+half)) * n + (j+half)];

    __syncthreads();

    int current_0 = d_Dist[i_n + j];
    int current_1 = d_Dist[i_half_n + j];
    int current_2 = d_Dist[i_n + (j+half)];
    int current_3 = d_Dist[i_half_n + (j+half)];

    #pragma unroll
    for (int k = 0; k < BS; k++) {
        current_0 = min(current_0, 
            shared[row_offset + il * BS + k] + 
            shared[col_offset + k * BS + jl]);
        current_1 = min(current_1, 
            shared[row_offset + (il+half) * BS + k] + 
            shared[col_offset + k * BS + jl]);
        current_2 = min(current_2, 
            shared[row_offset + il * BS + k] + 
            shared[col_offset + k * BS + (jl+half)]);
        current_3 = min(current_3, 
            shared[row_offset + (il+half) * BS + k] + 
            shared[col_offset + k * BS + (jl+half)]);
    }

    d_Dist[i_n + j] = current_0;
    d_Dist[i_half_n + j] = current_1;
    d_Dist[i_n + (j+half)] = current_2;
    d_Dist[i_half_n + (j+half)] = current_3;
}


int* input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    
    n_ori = n;
    n = ((n + BS - 1) / BS) * BS;

    int* Dist = (int*)malloc(n * n * sizeof(int));
    if (!Dist) {
        printf("Failed to allocate memory\n");
        exit(1);
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j && i < n_ori) {
                Dist[i * n + j] = 0;
            } else {
                Dist[i * n + j] = INF;
            }
        }
    }
    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * n + pair[1]] = pair[2];
    }
    fclose(file);
    return Dist;
}

void output(char* outFileName, int* Dist) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n_ori; ++i) {
        // for (int j = 0; j < n_ori; ++j) {
        //     if (Dist[i * n + j] >= INF) Dist[i * n + j] = INF;
        // }
        fwrite(&Dist[i * n], sizeof(int), n_ori, outfile);
    }
    fclose(outfile);
}

void block_FW(int* Dist) {
    int* d_Dist;
    

    size_t size = n * n * sizeof(int);
    cudaMalloc(&d_Dist, size);
    cudaMemcpy(d_Dist, Dist, size, cudaMemcpyHostToDevice);
    
    int round = ceil_div(n, BS);
    dim3 block_dim(TS, TS);
    dim3 grid_dim_p2(round - 1, 2);
    dim3 grid_dim_p3(round - 1, round - 1);
    for (int r = 0; r < round; ++r) {
        phase1_kernel<<<1, block_dim>>>(d_Dist, n, r);
        phase2_kernel<<<grid_dim_p2, block_dim>>>(d_Dist, n, r);
        phase3_kernel<<<grid_dim_p3, block_dim>>>(d_Dist, n, r);
    }
    cudaMemcpy(Dist, d_Dist, size, cudaMemcpyDeviceToHost);
    cudaFree(d_Dist);
}

int main(int argc, char* argv[]) {
    int* Dist = input(argv[1]);
    block_FW(Dist);
    output(argv[2], Dist);
    free(Dist);
    return 0;
}