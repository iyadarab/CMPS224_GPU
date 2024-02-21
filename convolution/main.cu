
#include "common.h"
#include "timer.h"

#define OUT_TILE_DIM 32

__constant__ float filter_c_[FILTER_DIM][FILTER_DIM];

__global__ void convolution_kernel(float* input, float* output, unsigned int width, unsigned int height) {
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    if (outRow < height && outCol < width) {
        float sum = 0.0f;
        for(int filterRow = 0; filterRow < FILTER_DIM; ++filterRow) {
            for(int filterCol = 0; filterCol < FILTER_DIM; ++filterCol) {
                int inRow = outRow - FILTER_RADIUS + filterRow;
                int inCol = outCol - FILTER_RADIUS + filterCol;
                if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                    sum += filter_c_[filterRow][filterCol]*input[inRow*width + inCol];
                }
            }
        }
        output[outRow*width + outCol] = sum;
    }
}

void convolution_gpu(float* input_d, float* output_d, unsigned int width, unsigned int height) {

    // Call kernel
    dim3 numThreadsPerBlock(OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 numBlocks((width + OUT_TILE_DIM - 1)/OUT_TILE_DIM, (height + OUT_TILE_DIM - 1)/OUT_TILE_DIM);
    convolution_kernel <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, width, height);

}

void convolution_cpu(float filter[][FILTER_DIM], float* input, float* output, unsigned int width, unsigned int height) {
    for (int outRow = 0; outRow < height; ++outRow) {
        for (int outCol = 0; outCol < width; ++outCol) {
            float sum = 0.0f;
            for(int filterRow = 0; filterRow < FILTER_DIM; ++filterRow) {
                for(int filterCol = 0; filterCol < FILTER_DIM; ++filterCol) {
                    int inRow = outRow - FILTER_RADIUS + filterRow;
                    int inCol = outCol - FILTER_RADIUS + filterCol;
                    if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        sum += filter[filterRow][filterCol]*input[inRow*width + inCol];
                    }
                }
            }
            output[outRow*width + outCol] = sum;
        }
    }
}

int main(int argc, char**argv) {

    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    float filter[FILTER_DIM][FILTER_DIM];
    unsigned int height = (argc > 1)?(atoi(argv[1])):4096;
    unsigned int width = (argc > 2)?(atoi(argv[2])):4096;
    float* input = (float*) malloc(width*height*sizeof(float));
    float* output_cpu = (float*) malloc(width*height*sizeof(float));
    float* output_gpu = (float*) malloc(width*height*sizeof(float));
    for (unsigned int filterRow = 0; filterRow < FILTER_DIM; ++filterRow) {
        for (unsigned int filterCol = 0; filterCol < FILTER_DIM; ++filterCol) {
            filter[filterRow][filterCol] = rand()*100.0/RAND_MAX;
        }
    }
    for (unsigned int row = 0; row < height; ++row) {
        for (unsigned int col = 0; col < width; ++col) {
            input[row*width + col] = rand()*100.0/RAND_MAX;
        }
    }

    // Compute on CPU
    startTime(&timer);
    convolution_cpu(filter, input, output_cpu, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // Allocate GPU memory
    startTime(&timer);
    float *input_d, *output_d;
    cudaMalloc((void**) &input_d, width*height*sizeof(float));
    cudaMalloc((void**) &output_d, width*height*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(input_d, input, width*height*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(filter_c_, filter, FILTER_DIM*FILTER_DIM*sizeof(float));
    copyFilterToGPU(filter);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Compute on GPU (without tiling)
    startTime(&timer);
    convolution_gpu(input_d, output_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time (without tiling)", GREEN);

    // Clear result
    cudaMemset(output_d, 0, width*height*sizeof(float));
    cudaDeviceSynchronize();

    // Compute on GPU (with tiling)
    startTime(&timer);
    convolution_tiled_gpu(input_d, output_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "GPU kernel time (with tiling)", GREEN);

    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(output_gpu, output_d, width*height*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

    // Verify result
    for (unsigned int row = 0; row < height; ++row) {
        for (unsigned int col = 0; col < width; ++col) {
            float diff = (output_cpu[row*width + col] - output_gpu[row*width + col])/output_cpu[row*width + col];
            const float tolerance = 0.00001;
            if(diff > tolerance || diff < -tolerance) {
                printf("Mismatch at row %u, col %u (CPU result = %e, GPU result = %e)\n", row, col, output_cpu[row*width + col], output_gpu[row*width + col]);
                exit(0);
            }
        }
    }

    // Free memory
    free(input);
    free(output_cpu);
    free(output_gpu);

    return 0;

}

