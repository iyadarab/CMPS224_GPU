
#include "common.h"

#include "timer.h"

__global__ void vecMax_kernel(double* a, double* b, double* c, unsigned int M) {

    // TODO
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < M) {
        if(a[i] >= b[i])
            c[i] = a[i];
        else
            c[i] = b[i];
    }
}

void vecMax_gpu(double* a, double* b, double* c, unsigned int M) {

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);

    // TODO
    double *a_d, *b_d, *c_d;
    cudaMalloc((void**) &a_d, M*sizeof(double));
    cudaMalloc((void**) &b_d, M*sizeof(double));
    cudaMalloc((void**) &c_d, M*sizeof(double));


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);

    // TODO
    cudaMemcpy(a_d, a, M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, M*sizeof(double), cudaMemcpyHostToDevice);


    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);

    // TODO
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = M/numThreadsPerBlock;
    vecMax_kernel <<< numBlocks, numThreadsPerBlock >>> (a_d, b_d, c_d, M);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);

    // TODO
    cudaMemcpy(c, c_d, M*sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);

    // TODO
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}

