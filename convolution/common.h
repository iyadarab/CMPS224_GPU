
#define FILTER_RADIUS 2
#define FILTER_DIM ((FILTER_RADIUS)*2 + 1)

void copyFilterToGPU(float filter[][FILTER_DIM]);

void convolution_tiled_gpu(float* input, float* output, unsigned int width, unsigned int height);

