#include <iostream>
#include <cuda_runtime.h>

#define TILE_DIM 16
#define FILTER_RADIUS 1
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)
#define MATRIX_WIDTH 1024

__constant__ float d_filter[FILTER_SIZE * FILTER_SIZE];

__global__ void convolutionTiled(float* input, float* output, int width, int height) {
    __shared__ float sharedMem[TILE_DIM + 2 * FILTER_RADIUS][TILE_DIM + 2 * FILTER_RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    // Load data into shared memory
    for (int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; ++dy) {
        for (int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; ++dx) {
            int sharedRow = ty + dy + FILTER_RADIUS;
            int sharedCol = tx + dx + FILTER_RADIUS;
            int inputRow = row + dy;
            int inputCol = col + dx;
            
            if (inputRow >= 0 && inputRow < height && inputCol >= 0 && inputCol < width) {
                sharedMem[sharedRow][sharedCol] = input[inputRow * width + inputCol];
            } else {
                sharedMem[sharedRow][sharedCol] = 0.0f;
            }
        }
    }

    __syncthreads();

    // Perform convolution
    float outputValue = 0.0f;
    if (row < height && col < width) {
        for (int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; ++dy) {
            for (int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; ++dx) {
                outputValue += sharedMem[ty + dy + FILTER_RADIUS][tx + dx + FILTER_RADIUS];
            }
        }
        output[row * width + col] = outputValue;
    }
}

void hostConvolution(float* input, float* output, int width, int height) {
    // Initialize filter to all 1s
    float h_filter[FILTER_SIZE * FILTER_SIZE];
    for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; ++i) {
        h_filter[i] = 1.0f;
    }

    cudaMemcpyToSymbol(d_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));

    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    convolutionTiled<<<dimGrid, dimBlock>>>(input, output, width, height);
}

void initializeMatrix(float* matrix, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    const int width = MATRIX_WIDTH;
    const int height = MATRIX_WIDTH;
    const int matrixSize = width * height;
    const int matrixBytes = matrixSize * sizeof(float);

    float *h_input = new float[matrixSize];
    float *h_output = new float[matrixSize];

    // Initialize the input matrix with random values
    initializeMatrix(h_input, width, height);

    float *d_input, *d_output;
    cudaMalloc(&d_input, matrixBytes);
    cudaMalloc(&d_output, matrixBytes);

    cudaMemcpy(d_input, h_input, matrixBytes, cudaMemcpyHostToDevice);

    hostConvolution(d_input, d_output, width, height);

    cudaMemcpy(h_output, d_output, matrixBytes, cudaMemcpyDeviceToHost);

    // Check the result here (optional)

    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_input;
    delete[] h_output;

    return 0;
}
