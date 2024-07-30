#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

__global__ void grayscale_kernel(unsigned char* input, unsigned char* output, int width, int height){
  const int channels = 3;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    int idx = row * width * channels + col * channels;
    int gray_ind = row * width + col;
    unsigned char r = input[idx];
    unsigned char g = input[idx + 1];  
    unsigned char b = input[idx + 2];
    output[gray_ind] = (unsigned char)(0.21f *r + 0.72f *g + 0.07 * b);
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}


torch::Tensor color_to_grayscale(torch::Tensor image) {
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kUInt8);

    const auto height = image.size(0);
    const auto width = image.size(1);

    auto output = torch::empty({height,width,1}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks(cdiv(width, threads_per_block.x), cdiv(height, threads_per_block.y));
    grayscale_kernel<<<number_of_blocks, threads_per_block>>>(
        image.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), width, height);

    return output;
}

