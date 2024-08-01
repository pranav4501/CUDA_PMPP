#include <iostream>
#include <cuda_runtime.h>

__global__ void matrixMulShared(float *d_C, const float *d_A, const float *d_B, int width) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    float value = 0;
    for (int i = 0; i < width; ++i) {
        float a = d_A[row * width + i];
        float b = d_B[i * width + col];
        value += a * b;
    }


    d_C[row * width + col] = value;
}

void matrixMultiplyHost(float *h_C, const float *h_A, const float *h_B, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            float sum = 0;
            for (int k = 0; k < width; ++k) {
                sum += h_A[i * width + k] * h_B[k * width + j];
            }
            h_C[i * width + j] = sum;
        }
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

int main() {
    const int WIDTH = 1024;
    const int MATRIX_SIZE = WIDTH * WIDTH;
    const int MATRIX_BYTES = MATRIX_SIZE * sizeof(float);

    float *h_A = new float[MATRIX_SIZE];
    float *h_B = new float[MATRIX_SIZE];
    float *h_C = new float[MATRIX_SIZE];
    float *h_C_ref = new float[MATRIX_SIZE];

    // Initialize matrices A and B
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device arrays
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, MATRIX_BYTES);
    cudaMalloc((void**)&d_B, MATRIX_BYTES);
    cudaMalloc((void**)&d_C, MATRIX_BYTES);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, MATRIX_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, MATRIX_BYTES, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(cdiv(WIDTH, dimBlock.x) , cdiv(WIDTH,dimBlock.y) );

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch the kernel
    matrixMulShared<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, WIDTH);

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Matrix multiplication using shared memory and tiling completed in " << milliseconds << " ms" << std::endl;

    // Copy result matrix C from device to host
    cudaMemcpy(h_C, d_C, MATRIX_BYTES, cudaMemcpyDeviceToHost);

    // Compute the reference solution on the host
    matrixMultiplyHost(h_C_ref, h_A, h_B, WIDTH);

    // Verify the result
    bool success = true;
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-4) {
            std::cout << "Mismatch at index " << i << ": " << h_C[i] << " != " << h_C_ref[i] << std::endl;
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "Matrix multiplication verified successfully." << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
