#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

__global__ void blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int blur_size){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int channel = threadIdx.z;
  int color_offset = channel*width*height;


  if (row < height && col < width) {
    int pix_val = 0;
    int pixxels = 0;

    for(int blur_row = -blur_size; blur_row <= blur_size; blur_row++){
      for(int blur_col = -blur_size; blur_col <= blur_size; blur_col++){
        int cur_blur_row = row + blur_row;
        int cur_blur_col = col + blur_col;

        if(cur_blur_row >= 0 && cur_blur_row < height && cur_blur_col >= 0 && cur_blur_col < width){
          pix_val += input[color_offset + cur_blur_row * width + cur_blur_col];
        }
        pixxels++;
      }
    }
    output[color_offset + row * width + col] = (unsigned char)(pix_val / pixxels);
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
} 

torch::Tensor blur_image(torch::Tensor image, int blur_size){
  assert(image.device().type() == torch::kCUDA);
  assert(image.dtype() == torch::kUInt8);

  const auto channels = image.size(0);
  const auto height = image.size(1); 
  const auto width = image.size(2);

  auto output = torch::empty_like(image);

  dim3 threads_per_block(16, 16, channels);
  dim3 number_of_blocks(cdiv(width, threads_per_block.x), cdiv(height, threads_per_block.y));

  blur_kernel<<<number_of_blocks, threads_per_block>>>(image.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), width, height, blur_size);

  return output;
}
