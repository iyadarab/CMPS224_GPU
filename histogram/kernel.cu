
#include "common.h"

#include "timer.h"

#define THREADS 256
#define COARSE_FACTOR 12

__global__ void histogram_private_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    // TODO
    __shared__ unsigned int bins_s[NUM_BINS];

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(threadIdx.x < 256){
        bins_s[threadIdx.x] = 0;
    }

    __syncthreads();

    if(i < width * height){
        atomicAdd(&bins_s[image[i]],1);
    }
    __syncthreads();

    if(threadIdx.x < 256 && bins[threadIdx.x] > 0){
        bins[threadIdx.x] += bins_s[threadIdx.x];
    }
}

void histogram_gpu_private(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    // TODO
    unsigned int numberofBlocks = ((width * height) + THREADS) / THREADS
    histogram_private_kernel <<< numberofBlocks, THREADS >>> (image_d, bins_d, width, height);
}

__global__ void histogram_private_coarse_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    // TODO
    __shared__ unsigned int bins_s[NUM_BINS];

    unsigned int i = blockDim.x * blockIdx.x * COARSE_FACTOR + threadIdx.x;

    if(threadIdx.x < 256){
        bins_s[threadIdx.x] = 0;
    }

    __syncthreads();

    if(i < width * height){
        for(unsigned int stride = 0 ; stride < COARSE_FACTOR; ++ stride){
            atomicAdd(&bins_s[image[i + stride * COARSE_FACTOR]],1);
        }
    }
    __syncthreads();

    if(threadIdx.x < 256 && bins[threadIdx.x] > 0){
        bins[threadIdx.x] += bins_s[threadIdx.x];
    }

}

void histogram_gpu_private_coarse(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    // TODO
    unsigned int numberofBlocks = (((width * height) + ((THREADS * COARSE_FACTOR -1) )) / (THREADS * COARSE_FACTOR))
    histogram_private_kernel <<< numberofBlocks, THREADS * COARSE_FACTOR >>> (image_d, bins_d, width, height);
}

