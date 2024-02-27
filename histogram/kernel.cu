
#include "common.h"

#include "timer.h"

#define THREADS_PER_BLOCK 256
#define COARSE_FACTOR 64

__global__ void histogram_private_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    // TODO
    __shared__ unsigned int bins_s[NUM_BINS];

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(threadIdx.x < NUM_BINS){
        bins_s[threadIdx.x] = 0;
    }

    __syncthreads();

    if(i < width * height){
        unsigned char character = image[i];
        atomicAdd(&bins_s[character],1);
    }
    __syncthreads();

    if(threadIdx.x < NUM_BINS){
        atomicAdd(&bins[threadIdx.x], bins_s[threadIdx.x]);
    }
}

void histogram_gpu_private(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    // TODO
    unsigned int numberofBlocks = ((width * height) +THREADS_PER_BLOCK) /THREADS_PER_BLOCK;
    histogram_private_kernel <<< numberofBlocks, THREADS_PER_BLOCK >>> (image_d, bins_d, width, height);
}

__global__ void histogram_private_coarse_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {

    // TODO
    __shared__ unsigned int bins_s[NUM_BINS];

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(threadIdx.x < NUM_BINS){
        bins_s[threadIdx.x] = 0;
    }

    __syncthreads();

    for(unsigned int stride = 0 ; stride < COARSE_FACTOR; ++stride){
            unsigned int index = i + stride * THREADS_PER_BLOCK;
            if(index < width * height){
                unsigned char character = image[index];
                atomicAdd(&bins_s[character],1);
            }
    }

    __syncthreads();

    if(threadIdx.x < NUM_BINS){
        atomicAdd(&bins[threadIdx.x], bins_s[threadIdx.x]);
    }

}

void histogram_gpu_private_coarse(unsigned char* image_d, unsigned int* bins_d, unsigned int width, unsigned int height) {

    // TODO
    unsigned int numberofBlocks = ((width * height) + (THREADS_PER_BLOCK * COARSE_FACTOR -1) ) / (THREADS_PER_BLOCK * COARSE_FACTOR);
    histogram_private_kernel <<< numberofBlocks, THREADS_PER_BLOCK>>> (image_d, bins_d, width, height);
}

