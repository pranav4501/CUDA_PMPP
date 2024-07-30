import os 
from pathlib import Path

from torch.utils.cpp_extension import load_inline
from torchvision.io import read_image, write_jpeg

def compile_extension():
  cuda_source = Path("/content/img_blr.cu").read_text()
  cpp_source = "torch::Tensor blur_image(torch::Tensor image, int blur_size);"

  os.makedirs("./cuda_build", exist_ok=True)
  blur_extension = load_inline(
      name="blur_extension",
      cpp_sources=cpp_source,
      cuda_sources=cuda_source,
      functions=["blur_image"],
      with_cuda=True,
      extra_cflags=["-O2"],
      build_directory="./cuda_build"
  )

  return blur_extension

def main():
  blur_module = compile_extension()

  input_image = read_image("real.jpg").contiguous().cuda()
  print(input_image.shape, input_image.dtype)
  output_image = blur_module.blur_image(input_image, 3)
  print(output_image.shape, output_image.dtype)
  write_jpeg(output_image.cpu(), "./output.jpeg")

if __name__ == "__main__":
  main
