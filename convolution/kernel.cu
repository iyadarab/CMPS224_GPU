
#include "common.h"

#include "timer.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float filter_c[FILTER_DIM][FILTER_DIM];

__global__ void convolution_tiled_kernel(float* input, float* output, unsigned int width, unsigned int height) {
    __shared__ float tile[IN_TILE_DIM][IN_TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * OUT_TILE_DIM + tx - FILTER_RADIUS;
    int by = blockIdx.y * OUT_TILE_DIM + ty - FILTER_RADIUS;

    int row = by;
    int col = bx;

    if (row >= 0 && row < height && col >= 0 && col < width) {
        tile[ty][tx] = input[row * width + col];
    } else {
        tile[ty][tx] = 0.0f;
    }

    __syncthreads();

    float sum = 0.0f;
    if (tx >= FILTER_RADIUS && tx < IN_TILE_DIM - FILTER_RADIUS && ty >= FILTER_RADIUS && ty < IN_TILE_DIM - FILTER_RADIUS) {
        for (int i = 0; i < FILTER_DIM; i++) {
            for (int j = 0; j < FILTER_DIM; j++) {
                sum += filter_c[i][j] * tile[ty + i - FILTER_RADIUS][tx + j - FILTER_RADIUS];
            }
        }

        if (row < height && col < width) {
            output[row * width + col] = sum;
        }
    }
}

void copyFilterToGPU(float filter[][FILTER_DIM]) {
    // Copy filter to constant memory
    cudaMemcpyToSymbol(filter_c, filter, FILTER_DIM * FILTER_DIM * sizeof(float));
}

void convolution_tiled_gpu(float* input_d, float* output_d, unsigned int width, unsigned int height) {
    // Call kernel
    dim3 numThreadsPerBlock(IN_TILE_DIM, IN_TILE_DIM);
    dim3 numBlocks((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    convolution_tiled_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, width, height);
}