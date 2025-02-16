#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <sys/time.h>

int B, N, d;
float *Q, *K, *V, *O;

__device__ float warpReduce(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

__global__ void forward_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V, 
    const int N, 
    const int d,
    const int TS, 
    const int BS, 
    const float softmax_scale,
    float* __restrict__ l, 
    float* __restrict__ m, 
    float* __restrict__ O
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int block_size = BS * d;
    int batch_off = bx * N * d;
    int lm_off = bx * N;
    extern __shared__ float smem[];
    float* qmem = smem;
    float* kmem = &smem[BS * d];
    float* vmem = &smem[BS * d * 2];
    float* smem_s = &smem[BS * d * 3];


    #pragma unroll 8
    for (int k = 0; k < TS; k++) {
        
        int block_off = batch_off + (block_size * k);
        kmem[ty * d + tx] = K[block_off + ty * d + tx];
        vmem[ty * d + tx] = V[block_off + ty * d + tx];

        if (d == 64) {
            kmem[ty * d + tx + 32] = K[block_off + (ty * d + tx + 32)];
            vmem[ty * d + tx + 32] = V[block_off + (ty * d + tx + 32)];
        }
        __syncthreads();
        
        for (int q = 0; q < TS; q++) {
            int block_off_q = batch_off + (block_size * q);
            qmem[ty * d + tx] = Q[block_off_q + ty * d + tx];
            if (d == 64) {
                qmem[ty * d + tx + 32] = Q[block_off_q + ty * d + tx + 32];
            }
            
            __syncthreads();

            float mij = -INFINITY;
            float lij = 0;

            for (int c = 0; c < BS; c++) {
                float qk = 0;
                qk += qmem[ty * d + tx] * kmem[c * d + tx];
                if (d == 64) {
                    qk += qmem[ty * d + tx + 32] * kmem[c * d + tx + 32];
                }
                
                qk = warpReduce(qk);

                if (tx == 0) {
                    qk *= softmax_scale;
                    smem_s[ty * BS + c] = qk;
                    mij = max(mij, qk);
                }
            }
            
            mij = __shfl_sync(0xffffffff, mij, 0);

            if (tx == 0) {
                #pragma unroll 16
                for (int c = 0; c < BS; c++) {
                    smem_s[ty * BS + c] = __expf(smem_s[ty * BS + c] - mij);
                    lij += smem_s[ty * BS + c];
                }
            }
            
            lij = __shfl_sync(0xffffffff, lij, 0);

            float qkv = 0;
            float qkv_2 = 0;

            #pragma unroll 16
            for (int c = 0; c < BS; c++) {
                qkv += smem_s[ty * BS + c] * vmem[c * d + tx];
                if (d == 64) {
                    qkv_2 += smem_s[ty * BS + c] * vmem[c * d + tx + 32];
                }
            }
            
            float m_old = m[lm_off + q * BS + ty];
            float l_old = l[lm_off + q * BS + ty];
            float m_new = max(m_old, mij);
            float l_new = (__expf(m_old - m_new) * l_old) + (__expf(mij - m_new) * lij);

            if (tx == 0) {
                m[lm_off + q * BS + ty] = m_new;
                l[lm_off + q * BS + ty] = l_new;
            }
            
            O[block_off_q + ty * d + tx] = 
                (1 / l_new) * 
                (
                    (l_old * __expf(m_old - m_new) * O[block_off_q + ty * d + tx]) + 
                    (__expf(mij - m_new) * qkv)
                );
            
            if (d == 64) {
                O[block_off_q + ty * d + tx + 32] = 
                    (1 / l_new) * 
                    (
                        (l_old * __expf(m_old - m_new) * O[block_off_q + ty * d + tx + 32]) + 
                        (__expf(mij - m_new) * qkv_2)
                    );
            }

            __syncthreads();
        }
        __syncthreads();
    }
}


void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");
    
    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);
    
    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));
    
    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");
    
    fwrite(O, sizeof(float), B * N * d, file);
    fclose(file);
}

int main(int argc, char *argv[]) {

    input(argv[1]);

    const int BS = 32;
    const int TS = ceil((float)N / BS);
    const float softmax_scale = 1.0f / sqrt(d);

    float *d_Q, *d_K, *d_V, *d_O;
    float *d_l, *d_m;
    
    size_t qkv_size = B * N * d * sizeof(float);
    size_t lm_size = B * N * sizeof(float);
    
    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMalloc(&d_O, qkv_size);
    cudaMalloc(&d_l, lm_size);
    cudaMalloc(&d_m, lm_size);

    cudaMemcpy(d_Q, Q, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, qkv_size, cudaMemcpyHostToDevice);

    float* h_m = new float[B * N];
    for (int i = 0; i < B * N; i++) {
        h_m[i] = -INFINITY;
    }
    cudaMemcpy(d_m, h_m, lm_size, cudaMemcpyHostToDevice);

    const int sram_size = (3 * BS * d * sizeof(float)) + (BS * BS * sizeof(float));
    dim3 grid_dim(B);
    dim3 block_dim(32, BS);

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        d_Q, d_K, d_V, N, d, TS, BS, softmax_scale,
        d_l, d_m, d_O
    );

    cudaMemcpy(O, d_O, qkv_size, cudaMemcpyDeviceToHost);

    output(argv[2]);

    return 0;
}