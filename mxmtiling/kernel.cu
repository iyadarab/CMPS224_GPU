// Author : Iyad Al Arab - iaa40@mail.aub.edu - 202203169
#include "common.h"

#include "timer.h"

#define TILE_DIM 32

__global__ void mm_tiled_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    // TODO
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for(unsigned int tile = 0; tile < N/TILE_DIM; ++tile) {
        // Load tile to shared memory and Adding boundary conditions inorder not to access invalid memory addresses
        unsigned int col_A = (tile * TILE_DIM) + threadIdx.x;

        if(row < N && col_A < K){
            A_s[threadIdx.y][threadIdx.x] = A[row*N + tile*TILE_DIM + threadIdx.x];
        } else {
            A_s[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        unsigned int row_B = (tile * TILE_DIM) + threadIdx.y;

        if(col < K && row_B < N){
            B_s[threadIdx.y][threadIdx.x] = B[(tile*TILE_DIM + threadIdx.y)*N + col];
        } else {
            B_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute with tile
        for(unsigned int i = 0; i < TILE_DIM; ++i) {
            sum += A_s[threadIdx.y][i]*B_s[i][threadIdx.x];
        }

        __syncthreads();
    }
    
    C[row*N + col] = sum;
}

void mm_gpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);

    // TODO
    float * d_A;
    float * d_B;
    float * d_C;
    cudaMalloc((void **) d_A, M * K * sizeof(float));
    cudaMalloc((void **) d_B, K * N * sizeof(float));
    cudaMalloc((void **) d_C, M * N * sizeof(float));

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);

    // TODO
    cudaMemcpy(d_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, K * N * sizeof(float), cudaMemcpyHostToDevice);


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);

    // TODO
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    mm_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);

    // TODO
    cudaMemcpy(d_C, M * N * sizeof(float));


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);

    // TODO
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}

