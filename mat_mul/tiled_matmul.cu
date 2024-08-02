#include <iostream>
#include <cuda_runtime.h> 

#define TILE_WIDTH 16

__global__ void matrixMulShared(float *d_C, const float *d_A, const float *d_B, int width) {

    __shared__ float a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float b[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float value = 0;

    int num_tiles = width / TILE_WIDTH;

    for (int i =0 ; i< num_tiles; ++i){
      a[ty][tx] = d_A[row*width + i*TILE_WIDTH + tx];
      b[ty][tx] = d_B[(i*TILE_WIDTH +ty)*width + col];
      __syncthreads();

      for( int k=0; k<TILE_WIDTH; ++k){
        value += a[ty][k] * b[k][tx];
      }
      __syncthreads();
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

int main(){
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

  cudaMemcpy(d_A, h_A, MATRIX_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, MATRIX_BYTES, cudaMemcpyHostToDevice);

  dim3 dimBlock(16, 16);
  dim3 dimGrid(cdiv(WIDTH, dimBlock.x) , cdiv(WIDTH,dimBlock.y) );

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  matrixMulShared<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, WIDTH);

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Matrix multiplication using shared memory and tiling completed in " << milliseconds << " ms" << std::endl;

  cudaMemcpy(h_C, d_C, MATRIX_BYTES, cudaMemcpyDeviceToHost);

  matrixMultiplyHost(h_C_ref, h_A, h_B, WIDTH);

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

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_C_ref;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
    
